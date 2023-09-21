import {config} from "dotenv";

config();

import {OpenAI} from "langchain/llms/openai";
import {ChatOpenAI} from "langchain/chat_models/openai";
import {HumanMessage} from "langchain/schema";
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

const llm = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0.9,
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
  const prompt = new PromptTemplate({
    template: "Tell me a {adjective} joke about the day {date}",
    inputVariables: ["adjective", "date"],
  });
};
