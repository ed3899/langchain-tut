import {config} from "dotenv";

config();
import {ChatOpenAI} from "langchain/chat_models/openai";

import {compile} from "html-to-text";
import {RecursiveUrlLoader} from "langchain/document_loaders/web/recursive_url";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import {HtmlToTextTransformer} from "langchain/document_transformers/html_to_text";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import * as R from "ramda";
import {HNSWLib} from "langchain/vectorstores/hnswlib";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import * as path from "path";
import {ConversationalRetrievalQAChain} from "langchain/chains";
import {BufferMemory} from "langchain/memory";

const recursiveUrlLoader = async () => {
  try {
    /**
     * Parses the html files recursively from the url and transforms them
     */
    const urlDocs = R.pipe(
      /**
       * Load the html files recursively from the url
       */
      async () =>
        await new RecursiveUrlLoader(
          "https://js.langchain.com//docs/get_started/introduction",
          {
            extractor: compile({wordwrap: 130}),
            maxDepth: 2,
            excludeDirs: ["https://js.langchain.com/docs/api/"],
            preventOutside: true,
          }
        ).load(),
      /**
       * Tranform the html files recursively from the url to text
       */
      async _urlDocs => {
        return await RecursiveCharacterTextSplitter.fromLanguage("html")
          .pipe(new HtmlToTextTransformer())
          .invoke(await _urlDocs);
      }
    );

    /**
     * Loads the pdf docs
     * @param pathToPdf
     * @returns
     */
    const pdfDocs = async (pathToPdf: string) =>
      await new PDFLoader(pathToPdf).load();

    /**
     * Saves the vector store to a file
     */
    const saveVectorStore = R.pipe(
      async (pathToSave: string) => {
        const mergedDocs = R.concat(
          await pdfDocs("./test-pdf.pdf"),
          await urlDocs()
        );

        return {
          mergedDocs,
          pathToSave,
        };
      },
      async pipedObject => {
        const embeddingsModel = new OpenAIEmbeddings({
          openAIApiKey: process.env.OPENAI_API_KEY,
          modelName: "text-embedding-ada-002",
        });

        return {
          ...(await pipedObject),
          embeddingsModel,
        };
      },
      async pipedObject => {
        const {mergedDocs, embeddingsModel} = await pipedObject;

        const vectorStore = await HNSWLib.fromDocuments(
          mergedDocs,
          embeddingsModel
        );

        return {
          ...(await pipedObject),
          vectorStore,
        };
      },
      async pipedObject => {
        const {vectorStore, pathToSave} = await pipedObject;
        await vectorStore.save(pathToSave);
      }
    );

    await saveVectorStore(path.join(process.cwd(), "vector-store"));
  } catch (error) {
    console.error(error);
  }
};

recursiveUrlLoader();

const bobTest = async () => {
  const CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT = `Given the following conversation and a follow up question, return the conversation history excerpt that includes any relevant context to the question if it exists and rephrase the follow up question to be a standalone question.
  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Your answer should follow the following format:
  \`\`\`
  Use the following pieces of context to answer the users question.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  ----------------
  <Relevant chat history excerpt as context here>
  Standalone question: <Rephrased question here>
  \`\`\`
  Your answer:`;

  const model = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-3.5-turbo",
    temperature: 0,
    maxRetries: 10,
    timeout: 20000,
  });

  const embeddingsModel = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-ada-002",
  });

  const dir = "vector-store";
  const loadedVectorStore = await HNSWLib.load(dir, embeddingsModel);

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    loadedVectorStore.asRetriever(),
    {
      memory: new BufferMemory({
        memoryKey: "chat_history",
        inputKey: "question",
        outputKey: "text",
        returnMessages: true,
      }),
      questionGeneratorChainOptions: {
        template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT,
      },
    }
  );

  const question =
    "I have a friend called Bob. He's 28 years old. He'd like to know who Eduardo is?";

  const res = await chain.call({
    question,
    chat_history: "",
  });

  console.log(res);

  let chatHistory = `${question}\n${res.text}`;

  const res2 = await chain.call({
    question: "How old is Bob?",
    chat_history: chatHistory,
  });

  console.log(res2);
  chatHistory = `${chatHistory}\n${res2.text}`;

  const res3 = await chain.call({
    question: "What is langchain?",
    chat_history: chatHistory,
  });

  console.log(res3);
};

// change in main

// bobTest();

// change

// change 2
