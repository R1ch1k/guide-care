// @ts-ignore - Custom dialog component
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/shadcn-io/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { NICEGuideline, NICEGraphNode } from "@/lib/types";

interface RuleExplanationModalProps {
    isOpen: boolean;
    onClose: () => void;
    path: string[]; // Array of Node IDs
    guideline: NICEGuideline;
}

export default function RuleExplanationModal({
    isOpen,
    onClose,
    path,
    guideline,
}: RuleExplanationModalProps) {
    // Map node IDs to their full objects for easy access
    const nodeMap = new Map<string, NICEGraphNode>(
        guideline.nodes.map((node) => [node.id, node])
    );

    // Create a map of edges for quick lookup: "from->to" => label
    const edgeMap = new Map<string, string>();
    guideline.edges.forEach((edge) => {
        edgeMap.set(`${edge.from}->${edge.to}`, edge.label || "");
    });

    const pathNodes = path
        .map((id) => nodeMap.get(id))
        .filter((node): node is NICEGraphNode => node !== undefined);

    return (
        <Dialog open={isOpen as boolean} onOpenChange={(open) => !open && onClose()}>
            <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle>Decision Path Explanation</DialogTitle>
                    <p className="sr-only">
                        A flowchart showing the sequence of decisions and rules applied by the chatbot.
                    </p>
                </DialogHeader>
                <ScrollArea className="flex-1 pr-4">
                    <div className="space-y-4 py-4">
                        {pathNodes.length === 0 ? (
                            <p className="text-gray-500 italic">No path information available.</p>
                        ) : (
                            pathNodes.map((node, index) => {
                                const nextNode = pathNodes[index + 1];
                                const edgeLabel = nextNode
                                    ? edgeMap.get(`${node.id}->${nextNode.id}`)
                                    : null;

                                return (
                                    <div key={`${node.id}-${index}`} className="relative">
                                        {/* Connector Line */}
                                        {index < pathNodes.length - 1 && (
                                            <div className="absolute left-6 top-10 bottom-[-16px] w-0.5 bg-gray-200" />
                                        )}

                                        <div className="flex items-start gap-4">
                                            {/* Icon/Badge */}
                                            <div
                                                className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center border-2 z-10 bg-white ${node.type === "action"
                                                    ? "border-blue-500 text-blue-600"
                                                    : "border-amber-500 text-amber-600"
                                                    }`}
                                            >
                                                {node.type === "action" ? (
                                                    <span className="text-lg">⚡</span>
                                                ) : (
                                                    <span className="text-lg">❓</span>
                                                )}
                                            </div>

                                            {/* Content */}
                                            <div className="flex-1 pt-1">
                                                <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
                                                    {node.type === "condition" ? "Condition Checked" : "Action Taken"}
                                                </p>
                                                <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                                                    <p className="text-gray-900 font-medium">{node.text}</p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Decision Label */}
                                        {index < pathNodes.length - 1 && edgeLabel && edgeLabel.toLowerCase() !== "next" && (
                                            <div className="ml-16 my-2 relative z-20">
                                                <span className={`text-sm font-bold uppercase tracking-wide ${edgeLabel.toLowerCase() === "yes"
                                                    ? "text-green-600"
                                                    : edgeLabel.toLowerCase() === "no"
                                                        ? "text-red-600"
                                                        : "text-gray-700"
                                                    }`}>
                                                    {edgeLabel}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                );
                            })
                        )}
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
