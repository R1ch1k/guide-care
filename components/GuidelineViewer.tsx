"use client";

import { useState } from "react";
import { NICEGuideline } from "@/lib/types";
import { Button } from "./ui/button";
import { X, FileJson, ListChecks, Copy, Check, Settings } from "lucide-react";

interface GuidelineViewerProps {
  guideline: NICEGuideline;
  onClose: () => void;
}

export default function GuidelineViewer({ guideline, onClose }: GuidelineViewerProps) {
  const [activeTab, setActiveTab] = useState<"rules" | "json" | "evaluators">("rules");
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    let content = "";
    if (activeTab === "rules") {
      content = guideline.rules.join("\n\n");
    } else if (activeTab === "json") {
      content = JSON.stringify(guideline, null, 2);
    } else if (activeTab === "evaluators" && guideline.condition_evaluators) {
      content = JSON.stringify(guideline.condition_evaluators, null, 2);
    }

    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{guideline.name}</h2>
            <p className="text-sm text-gray-500">{guideline.version}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b">
          <button
            onClick={() => setActiveTab("rules")}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "rules"
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            <ListChecks className="w-4 h-4" />
            IF-THEN Rules
          </button>
          <button
            onClick={() => setActiveTab("json")}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "json"
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            <FileJson className="w-4 h-4" />
            JSON Structure
          </button>
          <button
            onClick={() => setActiveTab("evaluators")}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === "evaluators"
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            <Settings className="w-4 h-4" />
            Evaluators
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === "rules" ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700">
                  Clinical Decision Rules ({guideline.rules.length})
                </h3>
                <Button
                  onClick={handleCopy}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      Copy All
                    </>
                  )}
                </Button>
              </div>
              <div className="space-y-3">
                {guideline.rules.map((rule, index) => (
                  <div
                    key={index}
                    className="bg-gray-50 border border-gray-200 rounded-lg p-4"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-semibold">
                        {index + 1}
                      </div>
                      <p className="text-sm text-gray-800 leading-relaxed flex-1">
                        {rule}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Graph Summary */}
              <div className="mt-6 pt-6 border-t">
                <h4 className="text-sm font-semibold text-gray-700 mb-3">
                  Decision Graph Summary
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <div className="text-2xl font-bold text-blue-700">
                      {guideline.nodes.filter(n => n.type === "condition").length}
                    </div>
                    <div className="text-xs text-blue-600 font-medium">
                      Decision Points
                    </div>
                  </div>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="text-2xl font-bold text-green-700">
                      {guideline.nodes.filter(n => n.type === "action").length}
                    </div>
                    <div className="text-xs text-green-600 font-medium">
                      Actions/Outcomes
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : activeTab === "json" ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700">
                  Complete JSON Structure
                </h3>
                <Button
                  onClick={handleCopy}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      Copy JSON
                    </>
                  )}
                </Button>
              </div>
              <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-auto text-xs">
                {JSON.stringify(guideline, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700">
                  Condition Evaluators
                  {guideline.condition_evaluators &&
                    ` (${Object.keys(guideline.condition_evaluators).length})`
                  }
                </h3>
                {guideline.condition_evaluators && (
                  <Button
                    onClick={handleCopy}
                    variant="outline"
                    size="sm"
                    className="gap-2"
                  >
                    {copied ? (
                      <>
                        <Check className="w-4 h-4" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4" />
                        Copy JSON
                      </>
                    )}
                  </Button>
                )}
              </div>

              {guideline.condition_evaluators ? (
                <div className="space-y-3">
                  {Object.entries(guideline.condition_evaluators).map(([nodeId, evaluator]) => {
                    const node = guideline.nodes.find(n => n.id === nodeId);
                    const getEvaluatorType = () => {
                      if ("type" in evaluator) {
                        return evaluator.type;
                      }
                      return "boolean";
                    };

                    const getEvaluatorBadgeColor = (type: string) => {
                      switch (type) {
                        case "bp_compare": return "bg-red-100 text-red-700 border-red-200";
                        case "bp_range": return "bg-pink-100 text-pink-700 border-pink-200";
                        case "age_compare": return "bg-purple-100 text-purple-700 border-purple-200";
                        case "numeric_compare": return "bg-blue-100 text-blue-700 border-blue-200";
                        case "or": return "bg-orange-100 text-orange-700 border-orange-200";
                        case "and": return "bg-green-100 text-green-700 border-green-200";
                        default: return "bg-gray-100 text-gray-700 border-gray-200";
                      }
                    };

                    const evalType = getEvaluatorType();

                    return (
                      <div
                        key={nodeId}
                        className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono font-semibold text-gray-900 bg-gray-100 px-2 py-1 rounded">
                              {nodeId}
                            </span>
                            <span className={`text-xs font-medium px-2 py-1 rounded border ${getEvaluatorBadgeColor(evalType)}`}>
                              {evalType.replace("_", " ").toUpperCase()}
                            </span>
                          </div>
                        </div>

                        {node && (
                          <p className="text-sm text-gray-600 mb-3 italic">
                            "{node.text}"
                          </p>
                        )}

                        <div className="bg-gray-50 rounded p-3">
                          <pre className="text-xs text-gray-800 font-mono">
                            {JSON.stringify(evaluator, null, 2)}
                          </pre>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Settings className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                  <h4 className="text-sm font-medium text-gray-900 mb-1">
                    No Evaluators Available
                  </h4>
                  <p className="text-sm text-gray-500">
                    This guideline was created before evaluator generation was implemented.
                    <br />
                    Try uploading a new PDF to generate evaluators automatically.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t p-4 bg-gray-50">
          <div className="flex items-center justify-between text-xs text-gray-600">
            <div>
              <span className="font-medium">Source:</span>{" "}
              <a
                href={guideline.citation_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                {guideline.citation}
              </a>
            </div>
            <Button onClick={onClose} variant="outline" size="sm">
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
