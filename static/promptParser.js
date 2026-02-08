/**
 * promptParser.js
 *
 * This module provides functions for parsing complex prompts and extracting task/subtask relationships.
 */

const SUBTASK_INDENT_REGEX = /^\s*-\s+/;

/**
 * Checks if a line is a subtask based on indentation.
 * @param {string} line - The line to check.
 * @returns {boolean} - True if the line is a subtask, false otherwise.
 */
function isSubtask(line) {
    return SUBTASK_INDENT_REGEX.test(line);
}

/**
 * Cleans a subtask line by removing the leading indentation characters.
 * @param {string} line - The subtask line to clean.
 * @returns {string} - The cleaned subtask line.
 */
function cleanSubtaskLine(line) {
    return line.replace(SUBTASK_INDENT_REGEX, '').trim();
}

/**
 * Parses a prompt into an array of tasks, identifying subtasks based on indentation or special characters.
 * @param {string} prompt - The input prompt string.
 * @returns {Array} - An array of task objects, each containing task text and subtasks.
 */
export function parsePromptIntoTasks(prompt) {
/**
     * Parses a prompt into an array of tasks, identifying subtasks based on indentation or special characters.
     * @param {string} prompt - The input prompt string.
     * @returns {Array} - An array of task objects, each containing task text and subtasks.
     */


    const tasks = [];
    let taskIdCounter = 0;

    // Regular expression to match task and subtask lines
    const taskRegex = /^(?:\s*\d+\.\s+)?([^\n]+)$/gm; // Matches any line, with or without leading numbers
    const subtaskRegex = /^\s*-\s+([^\n]+)$/gm; // Matches lines starting with a hyphen and whitespace

    let match;
    let currentTask = null;

    while ((match = taskRegex.exec(prompt)) !== null) {
        const line = match[1].trim();

        if (isSubtask(line)) {
            // This is a subtask
            if (currentTask) {
                currentTask.subtasks.push({
                    text: cleanSubtaskLine(line), // Remove the leading hyphen and space
                    status: 'pending'
                });
            }
        } else {
            // This is a main task
            currentTask = {
                id: taskIdCounter++,
                text: line,
                status: 'pending',
                result: '',
                subtasks: []
            };
            tasks.push(currentTask);
        }
    }

    return tasks;
}

/**
 * Parses a prompt into an array of tasks using NLP, identifying subtasks based on sentence structure.
 * @param {string} prompt - The input prompt string.
 * @returns {Array} - An array of task objects, each containing task text and subtasks.
 */
export function parsePromptIntoTasksNLP(prompt) {
    const nlp = window.nlp; // Assuming compromise is loaded globally or defined as window.nlp
    if (!nlp) {
        console.error("compromise NLP library not loaded.");
        return [];
    }

    try {
        const doc = nlp(prompt);
        return doc.sentences().json().map((sentence, index) => ({
            id: index,
            text: sentence.text,
            status: 'pending',
            result: '',
            subtasks: []
        }));
    } catch (error) {
        console.error("Error during NLP parsing:", error);
        return [];
    }
}