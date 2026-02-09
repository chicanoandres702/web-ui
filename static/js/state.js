export const state = {
    ws: null,
    stepCount: 0,
    tasks: [],
    taskGroups: [],
    currentFilter: 'all',
    currentEditTaskId: null,
    autoSaveEnabled: true,
    selectedTasks: new Set(),
    sessionStartTime: null,
    timerInterval: null,
    taskTemplates: {
        quiz: {
            name: "Complete Quiz",
            tasks: [
                { text: "Open quiz on learning platform", priority: "high", time: 2, tags: ["quiz", "homework"] },
                { text: "Read all questions carefully", priority: "high", time: 5, tags: ["quiz"] },
                { text: "Answer all questions", priority: "high", time: 30, tags: ["quiz"] },
                { text: "Review answers before submitting", priority: "medium", time: 10, tags: ["quiz"] },
                { text: "Submit quiz", priority: "high", time: 2, tags: ["quiz"] }
            ]
        },
        reading: {
            name: "Reading Assignment",
            tasks: [
                { text: "Locate reading material", priority: "high", time: 5, tags: ["reading", "homework"] },
                { text: "Read assigned chapters/pages", priority: "high", time: 45, tags: ["reading"] },
                { text: "Take notes on key points", priority: "medium", time: 20, tags: ["reading", "notes"] },
                { text: "Review and highlight important sections", priority: "low", time: 15, tags: ["reading"] },
                { text: "Complete reading comprehension questions", priority: "medium", time: 15, tags: ["reading", "homework"] }
            ]
        },
        paper: {
            name: "Research Paper",
            tasks: [
                { text: "Research topic and gather sources", priority: "high", time: 60, tags: ["paper", "research"] },
                { text: "Create outline", priority: "high", time: 30, tags: ["paper", "writing"] },
                { text: "Write introduction", priority: "medium", time: 30, tags: ["paper", "writing"] },
                { text: "Write body paragraphs", priority: "high", time: 90, tags: ["paper", "writing"] },
                { text: "Write conclusion", priority: "medium", time: 20, tags: ["paper", "writing"] },
                { text: "Add citations and bibliography", priority: "high", time: 30, tags: ["paper", "citations"] },
                { text: "Proofread and edit", priority: "medium", time: 30, tags: ["paper", "editing"] },
                { text: "Final review and submission", priority: "high", time: 15, tags: ["paper"] }
            ]
        },
        yellowdig: {
            name: "Yellowdig Post",
            tasks: [
                { text: "Review discussion prompt/topic", priority: "high", time: 10, tags: ["yellowdig", "discussion"] },
                { text: "Research and gather supporting information", priority: "medium", time: 20, tags: ["yellowdig", "research"] },
                { text: "Draft post (250-300 words)", priority: "high", time: 25, tags: ["yellowdig", "writing"] },
                { text: "Add relevant hashtags and citations", priority: "medium", time: 5, tags: ["yellowdig"] },
                { text: "Post and engage with 2 peer posts", priority: "medium", time: 20, tags: ["yellowdig", "engagement"] }
            ]
        },
        homework: {
            name: "General Homework",
            tasks: [
                { text: "Review assignment instructions", priority: "high", time: 5, tags: ["homework"] },
                { text: "Gather necessary materials", priority: "medium", time: 10, tags: ["homework"] },
                { text: "Complete main assignment tasks", priority: "high", time: 45, tags: ["homework"] },
                { text: "Double-check work for errors", priority: "medium", time: 15, tags: ["homework"] },
                { text: "Submit assignment", priority: "high", time: 5, tags: ["homework"] }
            ]
        },
        project: {
            name: "Project Workflow",
            tasks: [
                { text: "Define project scope and objectives", priority: "high", time: 30, tags: ["project", "planning"] },
                { text: "Create project timeline", priority: "high", time: 20, tags: ["project", "planning"] },
                { text: "Research and data collection", priority: "high", time: 90, tags: ["project", "research"] },
                { text: "Develop project deliverables", priority: "high", time: 120, tags: ["project", "development"] },
                { text: "Create presentation materials", priority: "medium", time: 60, tags: ["project", "presentation"] },
                { text: "Practice presentation", priority: "medium", time: 30, tags: ["project", "presentation"] },
                { text: "Final review and submit", priority: "high", time: 20, tags: ["project"] }
            ]
        }
    }
};

export function updateState(newState) {
    Object.assign(state, newState);
}