import {config} from "dotenv";

config();

import {OpenAI} from "langchain/llms/openai";
import {ChatOpenAI} from "langchain/chat_models/openai";
import {HumanMessage, LLMResult} from "langchain/schema";
import {StructuredOutputParser} from "langchain/output_parsers";
import {
  ChatPromptTemplate,
  ConditionalPromptSelector,
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import {
  BaseOutputParser,
  FormatInstructionsOptions,
} from "langchain/schema/output_parser";
import {Serialized} from "langchain/dist/load/serializable";

const llm = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0.9,
  maxRetries: 10,
  maxConcurrency: 5,
  cache: true,
  maxTokens: 25,
});

const chatModel = new ChatOpenAI();

const text =
  "What would be a good company name for a company that makes colorful socks?";

const prediction1 = async () => {
  const llmResult = await llm.predict(text);
  const chatModelResult = await chatModel.predict(text);

  console.log("Prediction 1");
  console.log(llmResult);
  console.log(chatModelResult);
};

const prediction2 = async () => {
  const messages = [new HumanMessage({content: text})];

  const llmResult = await llm.predictMessages(messages);
  const chatModelResult = await chatModel.predictMessages(messages);

  console.log("Prediction 2");
  console.log(llmResult);
  console.log(chatModelResult);
};

const prediction3 = async () => {
  const prompt = PromptTemplate.fromTemplate(
    "What is a good name for a company that makes {product}?"
  );

  const formattedPrompt = await prompt.format({
    product: "colorful socks",
  });

  console.log("Prediction 3");
  console.log(formattedPrompt);
};

const prediction4 = async () => {
  const template =
    "You are a helpful assistant that translates {input_language} to {output_language}.";
  const humanTemplate = "{text}";

  const chatPrompt = ChatPromptTemplate.fromMessages([
    ["system", template],
    ["human", humanTemplate],
  ]);

  const formattedChatPrompt = await chatPrompt.formatMessages({
    input_language: "English",
    output_language: "French",
    text: "I love programming.",
  });

  const llmResult = await llm.predictMessages(formattedChatPrompt);

  console.log(llmResult);
};

const prediction5 = async () => {
  /**
   * Parse the output of an LLM call to a comma-separated list.
   */
  class CommaSeparatedListOutputParser extends BaseOutputParser<string[]> {
    getFormatInstructions(
      options?: FormatInstructionsOptions | undefined
    ): string {
      throw new Error("Method not implemented.");
    }
    lc_namespace: string[] = [];
    async parse(text: string): Promise<string[]> {
      return text.split(",").map(item => item.trim());
    }
  }

  const template = `You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more.`;

  const humanTemplate = "{text}";

  /**
   * Chat prompt for generating comma-separated lists. It combines the system
   * template and the human template.
   */
  const chatPrompt = ChatPromptTemplate.fromMessages([
    ["system", template],
    ["human", humanTemplate],
  ]);

  const model = new ChatOpenAI({});
  const parser = new CommaSeparatedListOutputParser();

  const chain = chatPrompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    text: "colors",
  });

  console.log(result);
};

const prediction6 = async () => {
  const template =
    "You are a helpful assistant that translates {input_language} to {output_language}.";
  const systemMessagePrompt =
    SystemMessagePromptTemplate.fromTemplate(template);
  const humanTemplate = "{text}";
  const humanMessagePrompt =
    HumanMessagePromptTemplate.fromTemplate(humanTemplate);

  const chatPrompt = ChatPromptTemplate.fromMessages<{
    input_language: string;
    output_language: string;
    text: string;
  }>([systemMessagePrompt, humanMessagePrompt]);

  console.log(chatPrompt);
};

const partial1 = async () => {
  const prompt = new PromptTemplate({
    template: "{foo}{bar}",
    inputVariables: ["foo", "bar"],
  });

  const partialPrompt = await prompt.partial({
    foo: "foo",
  });

  const formattedPrompt = await partialPrompt.format({
    bar: "baz",
  });

  console.log(formattedPrompt);
};

const partial2 = async () => {
  const prompt = new PromptTemplate({
    template: "{foo}{bar}",
    inputVariables: ["bar"],
    partialVariables: {
      foo: "foo",
    },
  });

  const formattedPrompt = await prompt.format({
    bar: "baz",
  });

  console.log(formattedPrompt);
};

const partial3 = async () => {
  const getCurrentDate = () => {
    return new Date().toISOString();
  };

  const prompt = new PromptTemplate({
    template: "Tell me a {adjective} joke about the day {date}",
    inputVariables: ["adjective", "date"],
  });

  const partialPrompt = await prompt.partial({
    date: getCurrentDate,
  });

  const formattedPrompt = await partialPrompt.format({
    adjective: "funny",
  });
};

const prediction7 = async () => {
  const model = new OpenAI({temperature: 1});
  const controller = new AbortController();

  // Call `controller.abort()` somewhere to cancel the request.

  const res = await model.call(
    "What would be a good company name a company that makes colorful socks?",
    {signal: controller.signal}
  );

  console.log(res);
};

const prediction8 = async () => {
  const model = new OpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0.9,
    maxRetries: 10,
    maxConcurrency: 5,
    cache: true,
    maxTokens: 25,
    streaming: true,
  });

  const response = await model.call("Tell me a joke.", {
    callbacks: [
      {
        handleLLMNewToken(token: string) {
          console.log({token});
        },
      },
    ],
  });
  console.log(response);
};

const prediction9 = async () => {
  const model = new OpenAI({
    callbacks: [
      {
        handleLLMStart: async (llm: Serialized, prompts: string[]) => {
          console.log(JSON.stringify(llm, null, 2));
          console.log(JSON.stringify(prompts, null, 2));
        },
        handleLLMEnd: async (output: LLMResult) => {
          console.log(JSON.stringify(output, null, 2));
        },
        handleLLMError: async (err: Error) => {
          console.error(err);
        },
      },
    ],
  });

  await model.call(
    "What would be a good company name a company that makes colorful socks?"
  );
};

const prediction10 = async () => {
  // With a `StructuredOutputParser` we can define a schema for the output.
  const parser = StructuredOutputParser.fromNamesAndDescriptions({
    answer: "answer to the user's question",
    source: "source used to answer the user's question, should be a website.",
  });

  const formatInstructions = parser.getFormatInstructions();

  const prompt = new PromptTemplate({
    template:
      "Answer the users question as best as possible.\n{format_instructions}\n{question}",
    inputVariables: ["question"],
    partialVariables: {format_instructions: formatInstructions},
  });

  const model = new OpenAI({ temperature: 0 });
};

