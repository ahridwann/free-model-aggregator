// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// The rest of the 2x3 wire matrix: the starter only spoke the openai upstream
// wire. These translators cover the anthropic and gemini upstream wires for
// both client wires, in JSON and in server-sent events.
//
// Every function here is a pure function of its arguments: no clock, no
// randomness, fixed ids and timestamps, so the same provider answer gives
// the same bytes on a replay. The streamed translators that need a usage
// count up front (Anthropic's message_start, OpenAI's usage chunk) read the
// whole upstream stream first and then emit the translated sequence; the
// answers are small (one echo paragraph), so buffering costs nothing.
// ---------------------------------------------------------------------------

/** One parsed server-sent event frame: its event name, if any, and its data. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into its frames (event name + data payload each). */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A JSON value of one frame's data, or null when the frame is [DONE] or not JSON. */
function frameJson(frame: SseFrame): Json | null {
  const payload = frame.data.trim();
  if (payload === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The stop sequences of an OpenAI body: `stop` as a string or an array. */
function openAIStops(body: Json): string[] {
  const stop = body.stop;
  if (typeof stop === 'string') return [stop];
  if (Array.isArray(stop)) return stop.flatMap((s) => (typeof s === 'string' ? [s] : []));
  return [];
}

/** The stop sequences of an Anthropic body. */
function anthropicStops(body: Json): string[] {
  if (!Array.isArray(body.stop_sequences)) return [];
  return body.stop_sequences.flatMap((s) => (typeof s === 'string' ? [s] : []));
}

/** A data-URL image as its media type and base64 data, or null when it is not one. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] !== undefined && match[1] !== '' ? match[1] : 'image/png', data: match[2] ?? '' };
}

/** One OpenAI content part as Anthropic content blocks: text or an inline image. */
function openAIContentBlocks(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Json[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') blocks.push({ type: 'text', text: raw.text });
    else if (raw.type === 'image_url' && isRecord(raw.image_url) && typeof raw.image_url.url === 'string') {
      const url = raw.image_url.url;
      const data = dataUrlOf(url);
      if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } });
      else blocks.push({ type: 'image', source: { type: 'url', url } });
    }
  }
  return blocks;
}

/** Whether an Anthropic message holds only tool_result blocks (no text to echo). */
function toolResultOnly(message: Json): boolean {
  if (!Array.isArray(message.content) || message.content.length === 0) return false;
  return message.content.every((block) => isRecord(block) && block.type === 'tool_result');
}

/** One OpenAI function tool as an Anthropic tool. */
function openAIToolToAnthropic(raw: unknown): Json | null {
  if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return null;
  return {
    name: raw.function.name,
    description: typeof raw.function.description === 'string' ? raw.function.description : '',
    input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
  };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * prompt becomes `system`, image parts become image blocks, tool calls become tool_use blocks, tool
 * messages become tool_result blocks, and the output cap, temperature, top_p, stop sequences, tools,
 * tool choice and streaming pass through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let systemText = '';
  let lastUserText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'user') {
      const blocks = openAIContentBlocks(raw.content);
      if (blocks.length > 0) {
        messages.push({ role: 'user', content: blocks });
        const text = textOf(raw.content);
        if (text !== '') lastUserText = text;
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks = openAIContentBlocks(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const id = typeof call.id === 'string' ? call.id : '';
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        blocks.push({ type: 'tool_use', id, name, input: argumentsOf(call.function.arguments) });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (raw.role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: id, content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
  }
  // A body that ends on a tool result leaves the provider with no user text to echo, while the OpenAI
  // wire's echo is the last user message's text: re-state that text after it.
  const last = messages[messages.length - 1];
  if (last !== undefined && toolResultOnly(last) && lastUserText !== '') messages.push({ role: 'user', content: lastUserText });

  const out: Json = { model, messages };
  if (systemText !== '') out.system = systemText;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      const tool = openAIToolToAnthropic(raw);
      return tool === null ? [] : [tool];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function openAIFinishOfAnthropic(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages object as a Chat Completions object naming `model`. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const text = textOf(body.content);
  const calls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinishOfAnthropic(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * An Anthropic SSE stream (the whole text) as the OpenAI SSE stream: a role chunk, the text and tool
 * call deltas, the finish chunk, the usage chunk, then [DONE].
 */
export function anthropicStreamToOpenAIText(text: string, model: string): string {
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let content = '';
  const calls: { index: number; id: string; name: string; args: string }[] = [];
  for (const frame of parseSseFrames(text)) {
    const data = frameJson(frame);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : calls.length;
      calls.push({
        index,
        id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${index}`,
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        args: '',
      });
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') content += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const index = typeof data.index === 'number' ? data.index : -1;
        const call = calls.find((c) => c.index === index);
        if (call !== undefined) call.args += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-anthropic',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const out: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (content !== '') out.push(sseEvent(null, chunk({ content }, null)));
  for (const call of calls) {
    out.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index: call.index, id: call.id, type: 'function', function: { name: call.name, arguments: call.args } }] }, null),
      ),
    );
  }
  out.push(sseEvent(null, chunk({}, openAIFinishOfAnthropic(stopReason))));
  out.push(
    sseEvent(null, {
      id: 'chatcmpl-anthropic',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    }),
  );
  out.push('data: [DONE]\n\n');
  return out.join('');
}

// ---------------------------------------------------------------------------
// Gemini: the request body (contents + generationConfig) and the answers.
// ---------------------------------------------------------------------------

/** One OpenAI content part as Gemini parts: text, or an inline image / file reference. */
function openAIContentToGemini(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') parts.push({ text: raw.text });
    else if (raw.type === 'image_url' && isRecord(raw.image_url) && typeof raw.image_url.url === 'string') {
      const url = raw.image_url.url;
      const data = dataUrlOf(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
      else parts.push({ fileData: { mimeType: 'image/png', fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic content block as Gemini parts: text, image, tool_use or tool_result. */
function anthropicContentToGemini(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const raw of content) {
    if (!isRecord(raw)) continue;
    if (raw.type === 'text' && typeof raw.text === 'string') parts.push({ text: raw.text });
    else if (raw.type === 'image' && isRecord(raw.source)) {
      const source = raw.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: source.url } });
      }
    } else if (raw.type === 'tool_use') {
      parts.push({
        functionCall: { name: typeof raw.name === 'string' ? raw.name : '', args: isRecord(raw.input) ? raw.input : {} },
      });
    } else if (raw.type === 'tool_result') {
      parts.push({
        functionResponse: {
          name: typeof raw.tool_use_id === 'string' ? raw.tool_use_id : '',
          response: { content: toolResultText(raw.content) },
        },
      });
    }
  }
  return parts;
}

/** One tool definition as a Gemini function declaration. */
function toolToGeminiFunction(name: string, description: string, parameters: unknown): Json {
  return { name, description, parameters: isRecord(parameters) ? parameters : { type: 'object', properties: {} } };
}

/**
 * The Gemini request body from a system text, the contents, the function declarations, the output cap,
 * temperature, top_p, stop sequences and whether the client asked for JSON mode.
 */
function geminiRequestOf(
  systemText: string,
  contents: Json[],
  functions: Json[],
  maxTokens: number | null,
  temperature: number | undefined,
  topP: number | undefined,
  stops: string[],
  jsonMode: boolean,
): Json {
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (functions.length > 0) out.tools = [{ functionDeclarations: functions }];
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  let systemText = '';
  let lastUserText = '';
  const callNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'user') {
      const parts = openAIContentToGemini(raw.content);
      if (parts.length > 0) {
        contents.push({ role: 'user', parts });
        const text = textOf(raw.content);
        if (text !== '') lastUserText = text;
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const id = typeof call.id === 'string' ? call.id : '';
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (id !== '' && name !== '') callNames.set(id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      contents.push({
        role: 'function',
        parts: [
          {
            functionResponse: {
              name: callNames.get(id) ?? '',
              response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
            },
          },
        ],
      });
      continue;
    }
  }
  // A body that ends on a tool result leaves the provider with no user text to echo, while the OpenAI
  // wire's echo is the last user message's text: re-state that text after it.
  const last = contents[contents.length - 1];
  if (last !== undefined && last.role === 'function' && lastUserText !== '') contents.push({ role: 'user', parts: [{ text: lastUserText }] });

  const functions: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      functions.push(
        toolToGeminiFunction(
          raw.function.name,
          typeof raw.function.description === 'string' ? raw.function.description : '',
          raw.function.parameters,
        ),
      );
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  return geminiRequestOf(
    systemText,
    contents,
    functions,
    maxTokens,
    typeof body.temperature === 'number' ? body.temperature : undefined,
    typeof body.top_p === 'number' ? body.top_p : undefined,
    openAIStops(body),
    format === 'json_object' || format === 'json_schema',
  );
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  let lastUserText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const parts = anthropicContentToGemini(raw.content);
    if (parts.length === 0) continue;
    if (raw.role === 'assistant') {
      contents.push({ role: 'model', parts });
      continue;
    }
    contents.push({ role: 'user', parts });
    const text = textOf(raw.content);
    if (text !== '') lastUserText = text;
  }
  // A body that ends on a tool result leaves the provider with no user text to echo, while the
  // Anthropic wire's echo is the last user message's text: re-state that text after it.
  const last = contents[contents.length - 1];
  const lastParts = last !== undefined && Array.isArray(last.parts) ? last.parts : [];
  const lastIsFunctionOnly =
    last !== undefined && last.role === 'user' && lastParts.length > 0 && lastParts.every((p) => isRecord(p) && isRecord(p.functionResponse));
  if (lastIsFunctionOnly && lastUserText !== '') contents.push({ role: 'user', parts: [{ text: lastUserText }] });

  const functions: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      functions.push(toolToGeminiFunction(raw.name, typeof raw.description === 'string' ? raw.description : '', raw.input_schema));
    }
  }
  return geminiRequestOf(
    textOf(body.system),
    contents,
    functions,
    typeof body.max_tokens === 'number' ? body.max_tokens : null,
    typeof body.temperature === 'number' ? body.temperature : undefined,
    typeof body.top_p === 'number' ? body.top_p : undefined,
    anthropicStops(body),
    false,
  );
}

/** The text of a Gemini answer: its first candidate's parts, joined. */
function geminiTextOf(resp: Json): string {
  const candidates = Array.isArray(resp.candidates) ? resp.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** The usage of a Gemini answer, from its usageMetadata. */
function geminiUsageOf(resp: Json): { input: number; output: number } {
  const meta = isRecord(resp.usageMetadata) ? resp.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** The finish reason of a Gemini answer, or null when it names none. */
function geminiFinishOf(resp: Json): string | null {
  const candidates = Array.isArray(resp.candidates) ? resp.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return typeof first.finishReason === 'string' ? first.finishReason : null;
}

/** A Gemini finish reason as an OpenAI finish_reason. */
function openAIFinishOfGemini(finish: string | null): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finish reason as an Anthropic stop_reason. */
function anthropicStopOfGemini(finish: string | null): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** A Gemini generateContent answer as a Chat Completions object naming `model`. */
export function geminiToOpenAIResponse(resp: Json, model: string): Json {
  const usage = geminiUsageOf(resp);
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(resp) }, finish_reason: openAIFinishOfGemini(geminiFinishOf(resp)), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropicResponse(resp: Json, model: string): Json {
  const usage = geminiUsageOf(resp);
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiTextOf(resp) }],
    stop_reason: anthropicStopOfGemini(geminiFinishOf(resp)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One Gemini streamGenerateContent answer (the whole SSE text) read as text, usage and finish. */
function geminiStreamOf(text: string): { content: string; input: number; output: number; finish: string | null } {
  let content = '';
  let input = 0;
  let output = 0;
  let finish: string | null = null;
  for (const frame of parseSseFrames(text)) {
    const data = frameJson(frame);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const parts = isRecord(first.content) && Array.isArray(first.content.parts) ? first.content.parts : [];
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') content += part.text;
    if (typeof first.finishReason === 'string') finish = first.finishReason;
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { content, input, output, finish };
}

/**
 * A Gemini SSE stream as the OpenAI SSE stream: a role chunk, the text deltas, the finish chunk, the
 * usage chunk, then [DONE].
 */
export function geminiStreamToOpenAIText(text: string, model: string): string {
  const answer = geminiStreamOf(text);
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-gemini',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const out: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (answer.content !== '') out.push(sseEvent(null, chunk({ content: answer.content }, null)));
  out.push(sseEvent(null, chunk({}, openAIFinishOfGemini(answer.finish))));
  out.push(
    sseEvent(null, {
      id: 'chatcmpl-gemini',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * A Gemini SSE stream as the Anthropic SSE stream: message_start with the input tokens, one text
 * block, message_delta with the stop reason and output tokens, then message_stop.
 */
export function geminiStreamToAnthropicText(text: string, model: string): string {
  const answer = geminiStreamOf(text);
  const out: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_gemini',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  if (answer.content !== '') {
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.content } }));
  }
  out.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOfGemini(answer.finish), stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}
