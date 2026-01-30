// @ts-ignore - Custom dialog component
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/shadcn-io/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

interface SampleInputModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function SampleInputModal({ isOpen, onClose }: SampleInputModalProps) {
    const samples = [
        {
            title: "Hypertensive Emergency",
            input: "Patient has clinic BP 190/125 mmHg and shows signs of papilloedema.",
            expected: "Refer for same-day specialist assessment (Rule 1)",
            color: "bg-red-50 border-red-200",
        },
        {
            title: "Urgent Treatment Start",
            input: "BP is 185/122 mmHg with signs of target-organ damage. No emergency symptoms.",
            expected: "Start antihypertensive drug treatment immediately (Rule 2)",
            color: "bg-orange-50 border-orange-200",
        },
        {
            title: "Diagnosis Confirmation",
            input: "Clinic BP is 150/95 mmHg.",
            expected: "Offer ambulatory blood pressure monitoring (ABPM) (Rule 5)",
            color: "bg-blue-50 border-blue-200",
        },
        {
            title: "Treatment Threshold (Under 80)",
            input: "Hypertension confirmed. Patient is 65 years old with QRISK score of 15%.",
            expected: "Offer antihypertensive drug treatment (Rule 8)",
            color: "bg-green-50 border-green-200",
        },
        {
            title: "Treatment Choice (Step 1)",
            input: "Starting treatment for 45 year old white male.",
            expected: "Offer ACE inhibitor or ARB (Rule 13)",
            color: "bg-purple-50 border-purple-200",
        },
    ];

    return (
        <Dialog open={isOpen as boolean} onOpenChange={(open) => !open && onClose()}>
            <DialogContent className="max-w-2xl">
                <DialogHeader>
                    <DialogTitle>Sample Inputs & Expected Outcomes</DialogTitle>
                    <p className="sr-only">
                        A list of sample clinical scenarios and their expected outcomes according to NICE guidelines.
                    </p>
                </DialogHeader>
                <ScrollArea className="max-h-[60vh] pr-4">
                    <div className="space-y-4">
                        <p className="text-sm text-gray-600">
                            Try these examples to see how the chatbot applies NICE guidelines to different clinical scenarios.
                        </p>
                        <div className="grid gap-4">
                            {samples.map((sample, index) => (
                                <div
                                    key={index}
                                    className={`p-4 rounded-lg border ${sample.color} transition-all hover:shadow-sm`}
                                >
                                    <h3 className="font-semibold text-gray-900 mb-2">{sample.title}</h3>
                                    <div className="space-y-2">
                                        <div>
                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Input</span>
                                            <p className="text-sm text-gray-800 mt-0.5">{sample.input}</p>
                                        </div>
                                        <div>
                                            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Expected Result</span>
                                            <p className="text-sm font-medium text-gray-900 mt-0.5">{sample.expected}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}
