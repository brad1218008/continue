import { ChatCompletionCreateParams } from "@continuedev/openai-adapters";

import { ChatMessage, CompletionOptions, LLMOptions } from "../../index.js";
import { renderChatMessage } from "../../util/messageContent.js";
import { BaseLLM } from "../index.js";
import { toChatBody } from "../openaiTypeConverters.js";
import { streamSse } from "../stream.js";

const MODEL = ["gpt4", "gpt4o-mini"];

class Fujitsu extends BaseLLM {
  static providerName = "fujitsu";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://api.ai-service.global.fujitsu.com/ai-foundation/chat-ai/",
  };

  protected _getHeaders() {
    return {
      "Content-Type": "application/json",
      "api-key": this.apiKey ?? "",
    };
  }

  protected _convertArgs(
    options: CompletionOptions,
    messages: ChatMessage[],
  ): ChatCompletionCreateParams {
    const finalOptions = toChatBody(messages, options);

    finalOptions.stop = options.stop?.slice(0, this.getMaxStopWords());

    finalOptions.prediction = undefined;

    return finalOptions;
  }

  protected getMaxStopWords(): number {
    const url = new URL(this.apiBase!);

    if (this.maxStopWords !== undefined) {
      return this.maxStopWords;
    } else if (url.host === "api.deepseek.com") {
      return 16;
    } else if (
      url.port === "1337" ||
      url.host === "api.openai.com" ||
      url.host === "api.groq.com" ||
      this.apiType === "azure"
    ) {
      return 4;
    } else {
      return Infinity;
    }
  }

  protected async *_streamComplete(
    prompt: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    for await (const chunk of this._streamChat(
      [{ role: "user", content: prompt }],
      signal,
      options,
    )) {
      yield renderChatMessage(chunk);
    }
  }

  protected async *_streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const args: any = this._convertArgs(options, messages);

    const body = JSON.stringify({
      ...args,
      stream: true,
      ...this.extraBodyProperties(),
    });

    this.requestOptions = {
      ...this.requestOptions,
      verifySsl: false,
    };
    
    const response = await this.fetch(new URL(this.modelToPath(), this.apiBase), {
      method: "POST",
      headers: this._getHeaders(),
      body: body,
      signal,
    });

    for await (const value of streamSse(response)) {
      if (value.choices?.[0]?.delta?.content && value.finish_reason !== "stop") {
        const chatMessage: ChatMessage = {
          role: "assistant",
          content: value.choices[0].delta.content,
        };
        yield chatMessage;
      }
    }
    return;
  }
  
  protected extraBodyProperties(): Record<string, any> {
    return {};
  }

  private modelToPath(): string {
    if(this.model === "gpt-4o") {
      return "gpt4";
    } else {
      return "gpt4o-mini";
    }
  }
}

export default Fujitsu;
