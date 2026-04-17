package com.articulate.sigma.utils;

import com.articulate.nlp.KBLite;
import com.articulate.sigma.ErrRec;
import com.articulate.sigma.KifFileChecker;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * SumoChecker reads a .kif or .txt file and checks it against the SUMO KB.
 *
 * Usage:
 *   SumoChecker <file> [--check_lbl] [--check_syntax] [--print_not_found N] [--print_syntax_errors N]
 *
 * Flags:
 *   --check_lbl              Line-by-line term coverage check against KBLite.
 *   --check_syntax           ANTLR syntax check; reports per-line error statistics.
 *   --print_not_found N      Print the first N unique terms not found in KBLite.
 *   --print_syntax_errors N  Print the first N syntax errors.
 */
public class SumoChecker {

    // KIF logical keywords and connectives to ignore when extracting terms
    private static final Set<String> KEYWORDS = new HashSet<>(Arrays.asList(
            "subclass", "relation", "subrelation", "instance",
            "and", "or", "xor", "exists", "forall", "=>",
            "not", "Now", "equal"
    ));

    public static void main(String[] args) {
        String inputFile = null;
        boolean checkLbl = false;
        boolean checkSyntax = false;
        int printNotFound = -1;
        int printSyntaxErrors = -1;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--check_lbl":
                    checkLbl = true;
                    break;
                case "--check_syntax":
                    checkSyntax = true;
                    break;
                case "--print_not_found":
                    if (i + 1 < args.length) {
                        try {
                            printNotFound = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Error: --print_not_found requires an integer argument.");
                            System.exit(1);
                        }
                    } else {
                        System.err.println("Error: --print_not_found requires an integer argument.");
                        System.exit(1);
                    }
                    break;
                case "--print_syntax_errors":
                    if (i + 1 < args.length) {
                        try {
                            printSyntaxErrors = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Error: --print_syntax_errors requires an integer argument.");
                            System.exit(1);
                        }
                    } else {
                        System.err.println("Error: --print_syntax_errors requires an integer argument.");
                        System.exit(1);
                    }
                    break;
                default:
                    if (!args[i].startsWith("--")) {
                        inputFile = args[i];
                    } else {
                        System.err.println("Warning: Unknown flag '" + args[i] + "' (ignoring).");
                    }
                    break;
            }
        }

        if (inputFile == null) {
            printUsage();
            System.exit(1);
        }

        File f = new File(inputFile);
        if (!f.exists() || !f.isFile()) {
            System.err.println("Error: File not found or is not a regular file: " + inputFile);
            System.exit(1);
        }

        if (!checkLbl && !checkSyntax) {
            System.err.println("Error: No processing mode specified. Use --check_lbl and/or --check_syntax.");
            printUsage();
            System.exit(1);
        }

        if (checkSyntax) {
            runSyntaxCheck(inputFile, printSyntaxErrors);
        }

        if (checkLbl) {
            KBLite kb = new KBLite("SUMO");
            processLineByLine(inputFile, kb, printNotFound);
        }
    }

    /**
     * Extracts SUMO terms from a single line of KIF/text.
     * Strips quoted strings, parentheses, variables (?...), keywords, and whitespace.
     */
    static List<String> extractTerms(String line) {
        List<String> terms = new ArrayList<>();

        // Remove all quoted strings (including their contents)
        line = line.replaceAll("\"(?:[^\"\\\\]|\\\\.)*\"", " ");

        // Remove parentheses (they are structural, not terms)
        line = line.replace("(", " ").replace(")", " ");

        // Tokenize on whitespace
        String[] tokens = line.trim().split("\\s+");
        for (String token : tokens) {
            if (token.isEmpty()) continue;
            if (token.startsWith("?")) continue;      // variable
            if (KEYWORDS.contains(token)) continue;   // logical keyword
            if (token.matches("\\d+")) continue;      // pure number
            if (token.startsWith("UNK_")) continue;   // unknown token marker
            terms.add(token);
        }
        return terms;
    }

    private static void runSyntaxCheck(String filePath, int printSyntaxErrors) {
        String contents;
        String[] fileLines;
        try {
            byte[] bytes = Files.readAllBytes(Paths.get(filePath));
            contents = new String(bytes);
            fileLines = contents.split("\n", -1);
        } catch (IOException e) {
            System.err.println("Error reading file for syntax check: " + filePath);
            e.printStackTrace();
            return;
        }

        List<ErrRec> msgs = new ArrayList<>();
        KifFileChecker.CheckSyntaxErrors(contents, filePath, msgs);

        // Collect the set of unique line numbers that have at least one error
        Set<Integer> errorLines = new HashSet<>();
        for (ErrRec err : msgs) {
            errorLines.add(err.line);
        }

        int totalLines = fileLines.length;
        int linesWithErrors = errorLines.size();
        double pctErrors = totalLines > 0 ? 100.0 * linesWithErrors / totalLines : 0.0;

        System.out.println("\n=== Syntax Check Statistics ===");
        System.out.printf("Total lines:                          %d%n",      totalLines);
        System.out.printf("Lines with syntax errors:             %d%n",      linesWithErrors);
        System.out.printf("Percent of lines with syntax errors:  %.2f%%%n",  pctErrors);

        if (printSyntaxErrors > 0) {
            System.out.println("\n=== First " + printSyntaxErrors + " syntax errors ===");
            int count = Math.min(printSyntaxErrors, msgs.size());
            for (int i = 0; i < count; i++) {
                ErrRec err = msgs.get(i);
                System.out.println("  " + err.toString());
                int lineIdx = err.line - 1; // ErrRec lines are 1-based
                if (lineIdx >= 0 && lineIdx < fileLines.length) {
                    System.out.println("    > " + fileLines[lineIdx]);
                }
            }
        }
    }

    private static void processLineByLine(String filePath, KBLite kb, int printNotFound) {
        int totalLines = 0;
        int linesWithUnknownTerms = 0;
        int totalTermCount = 0;
        int totalUnknownTermCount = 0;

        // LinkedHashSet/Map preserves first-seen insertion order
        Set<String> uniqueTerms = new LinkedHashSet<>();
        // Maps unknown term -> the line it was first found on
        Map<String, String> unknownTermToLine = new LinkedHashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                totalLines++;
                List<String> lineTerms = extractTerms(line);
                boolean lineHasUnknown = false;

                for (String term : lineTerms) {
                    totalTermCount++;
                    uniqueTerms.add(term);

                    if (!kb.terms.contains(term)) {
                        totalUnknownTermCount++;
                        unknownTermToLine.putIfAbsent(term, line);
                        lineHasUnknown = true;
                    }
                }

                if (lineHasUnknown) {
                    linesWithUnknownTerms++;
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + filePath);
            e.printStackTrace();
            return;
        }

        int uniqueTermCount = uniqueTerms.size();
        int uniqueUnknownCount = unknownTermToLine.size();

        double pctUniqueUnknown  = uniqueTermCount > 0  ? 100.0 * uniqueUnknownCount       / uniqueTermCount  : 0.0;
        double pctAllUnknown     = totalTermCount > 0   ? 100.0 * totalUnknownTermCount     / totalTermCount   : 0.0;
        double pctLinesUnknown   = totalLines > 0       ? 100.0 * linesWithUnknownTerms     / totalLines       : 0.0;

        System.out.println("\n=== SumoChecker Statistics ===");
        System.out.printf("Total unique terms:                              %d%n",     uniqueTermCount);
        System.out.printf("Percent unique terms not in KBLite:             %.2f%%%n",  pctUniqueUnknown);
        System.out.printf("Total terms (all occurrences):                  %d%n",     totalTermCount);
        System.out.printf("Percent of all terms not found in SUMO:         %.2f%%%n",  pctAllUnknown);
        System.out.printf("Total lines:                                    %d%n",     totalLines);
        System.out.printf("Lines with at least one unknown term:           %d%n",     linesWithUnknownTerms);
        System.out.printf("Percent of lines with at least one unknown term: %.2f%%%n", pctLinesUnknown);

        if (printNotFound > 0) {
            System.out.println("\n=== First " + printNotFound + " unique terms not found in KBLite ===");
            int count = 0;
            for (Map.Entry<String, String> entry : unknownTermToLine.entrySet()) {
                if (count >= printNotFound) break;
                System.out.println("  " + entry.getKey() + "\t| " + entry.getValue());
                count++;
            }
        }
    }

    private static void printUsage() {
        System.err.println("Usage: SumoChecker <file.kif|file.txt> [--check_lbl] [--check_syntax] [--print_not_found N] [--print_syntax_errors N]");
        System.err.println("  --check_lbl              Check each line for SUMO term coverage.");
        System.err.println("  --check_syntax           Run ANTLR syntax check and report statistics.");
        System.err.println("  --print_not_found N      Print the first N unique terms not found in KBLite.");
        System.err.println("  --print_syntax_errors N  Print the first N syntax errors.");
    }
}
