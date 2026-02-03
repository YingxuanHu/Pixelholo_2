
import React from 'react';
import { StepStatus } from '../types';

interface StepCardProps {
  stepNumber: number;
  title: string;
  description: string;
  status: StepStatus;
  statusSteps?: string[];
  statusNote?: string;
  progress?: number | null;
  activeStepIndex?: number | null;
  statusContent?: React.ReactNode;
  showStepsWithContent?: boolean;
  children: React.ReactNode;
  isActive?: boolean;
}

const StepCard: React.FC<StepCardProps> = ({ 
  stepNumber, 
  title, 
  description, 
  status, 
  statusSteps = [],
  statusNote,
  progress = null,
  activeStepIndex = null,
  statusContent,
  showStepsWithContent = false,
  children,
  isActive = false
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'running': return 'bg-amber-500';
      case 'done': return 'bg-teal-600';
      case 'error': return 'bg-red-600';
      default: return 'bg-slate-300';
    }
  };

  const getBadgeText = () => {
    switch (status) {
      case 'running': return 'Active Process';
      case 'done': return 'Phase Complete';
      case 'error': return 'Issue Detected';
      default: return 'Waiting...';
    }
  };

  return (
    <div className={`relative bg-white border-2 rounded-[28px] overflow-hidden transition-all duration-500 min-h-[620px] ${isActive ? 'border-teal-600/20 shadow-2xl shadow-teal-600/5' : 'border-slate-100 opacity-60 pointer-events-none'}`}>
      <div className="px-10 pt-12 pb-14">
        <div className="flex items-start justify-between mb-8">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest text-white ${getStatusColor()}`}>
                {getBadgeText()}
              </span>
              <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest">Step 0{stepNumber}</span>
            </div>
            <h2 className="text-5xl font-bold text-slate-900 mb-4">{title}</h2>
            <p className="text-slate-500 text-xl font-medium leading-relaxed max-w-2xl">
              {description}
            </p>
          </div>
          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center text-white font-black text-3xl shadow-xl ${getStatusColor()}`}>
            {stepNumber}
          </div>
        </div>

        {status === 'running' && statusContent && (
          <div className="mb-8">
            {showStepsWithContent && statusSteps.length > 0 ? (
              <div className="grid md:grid-cols-[1.1fr_1fr] gap-6">
                <div>{statusContent}</div>
                <div className="bg-white border border-slate-100 rounded-2xl p-5 shadow-sm">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Expected stages</p>
                  <ol className="mt-4 space-y-2 text-sm text-slate-600">
                    {statusSteps.map((step, index) => (
                      <li key={`${step}-${index}`} className="flex items-center gap-3">
                        <span
                          className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold ${
                            activeStepIndex !== null && index < activeStepIndex
                              ? 'bg-teal-600 text-white'
                              : activeStepIndex === index
                                ? 'bg-amber-500 text-white'
                                : 'bg-slate-100 text-slate-500'
                          }`}
                        >
                          {index + 1}
                        </span>
                        <span
                          className={`${
                            activeStepIndex !== null && index === activeStepIndex
                              ? 'text-slate-900 font-semibold'
                              : ''
                          }`}
                        >
                          {step}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            ) : (
              statusContent
            )}
          </div>
        )}

        {status === 'running' && !statusContent && statusSteps.length > 0 && (
          <div className="grid md:grid-cols-[1.1fr_1fr] gap-6 mb-8">
            <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5 shadow-sm">
              <p className="text-[10px] font-bold uppercase tracking-widest text-amber-600">Live Pipeline</p>
              <p className="text-sm text-amber-900 font-semibold mt-2">
                Processing in real time. Outputs stream as soon as each stage completes.
              </p>
              {statusNote && (
                <p className="text-xs text-amber-700 mt-2">{statusNote}</p>
              )}
              {activeStepIndex !== null && statusSteps[activeStepIndex] && (
                <div className="mt-3 text-xs font-semibold text-amber-900">
                  Now: {statusSteps[activeStepIndex]}
                </div>
              )}
              {progress !== null && (
                <div className="mt-3 text-xs font-semibold text-amber-900">
                  Progress: {Math.round(progress * 100)}%
                </div>
              )}
              <div className="mt-4 flex items-center gap-3 text-[10px] font-bold uppercase tracking-widest text-amber-700">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-500 opacity-70"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-600"></span>
                </span>
                Running now
              </div>
            </div>
            <div className="bg-white border border-slate-100 rounded-2xl p-5 shadow-sm">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Expected stages</p>
              <ol className="mt-4 space-y-2 text-sm text-slate-600">
                {statusSteps.map((step, index) => (
                  <li key={`${step}-${index}`} className="flex items-center gap-3">
                    <span
                      className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold ${
                        activeStepIndex !== null && index < activeStepIndex
                          ? 'bg-teal-600 text-white'
                          : activeStepIndex === index
                            ? 'bg-amber-500 text-white'
                            : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {index + 1}
                    </span>
                    <span
                      className={`${
                        activeStepIndex !== null && index === activeStepIndex
                          ? 'text-slate-900 font-semibold'
                          : ''
                      }`}
                    >
                      {step}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}
        
        <div className="bg-white rounded-2xl">
          {children}
        </div>
      </div>
      
    </div>
  );
};

export default StepCard;
