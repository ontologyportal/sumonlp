import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;

public class LlamaMTrans {

    // Hardcoded path to Ollama binary
    // private static final String OLLAMA_PATH = "/home/jarrad.singley/Programs/llama/bin/ollama";

    public static void main(String[] args) {
        // Ensure correct number of arguments
        if (args.length != 2) {
            System.out.println("Usage: java LlamaMD <inputFile> <outputFile>");
            return;
        }

        String inputFile = args[0];  // Use the first argument as the input file path
        String outputFile = args[1]; // Use the second argument as the output file path

        processFile(inputFile, outputFile);
    }

    public static void processFile(String inputFile, String outputFile) {
        try (BufferedReader br = new BufferedReader(new FileReader(inputFile));
             FileWriter writer = new FileWriter(outputFile)) {

            String sentence;
            while ((sentence = br.readLine()) != null) {

                if (sentence.startsWith("1")) {
                    // Remove the leading "1" and any tab character from the sentence
                    sentence = sentence.substring(1).replace("\t", "");
                    String result = askOllama("The following sentence contains metaphorical content:  " + sentence + "  Translate the sentence so that no metaphorical expressions are present. Make sure there is no figurative language, make the sentence as plain and literal as possible. MOST IMPORTANTLY, respond with ONLY the translated sentence.");
                    System.out.println("-----------------------\n");
                    System.out.println("result: " + result);
                    System.out.println("-----------------------\n");
                    result = result.replace("\n", "");
                    writer.write(result + "\n");
                    
                } else {
                    sentence = sentence.substring(1).replace("\t", "");
                    writer.write(sentence + "\n");
                }
                // Call Ollama with each sentence and get the result
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String askOllama(String sentence) {
        StringBuilder result = new StringBuilder();
        try {
            // Command to run Ollama with the hardcoded path
            ProcessBuilder processBuilder = new ProcessBuilder("ollama", "run", "llama3.2");

            Process process = processBuilder.start();
            
            // Get input stream to write the sentence to Ollama
            process.getOutputStream().write((sentence + "\n").getBytes());
            process.getOutputStream().flush();
            process.getOutputStream().close();

            // Capture the output from Ollama
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line).append("\n");  // Collect the response
            }

            process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        System.out.print(".");
        return result.toString();
    }
}
