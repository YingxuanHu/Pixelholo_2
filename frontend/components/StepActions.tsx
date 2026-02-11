import React from 'react';
import MobileBottomBar from '../mobile/MobileBottomBar';
import { useIsMobile } from '../mobile/useIsMobile';

type StepActionsProps = {
  activeStep: number;
  isBusy: boolean;
  canProceedTo: (step: number) => boolean;
  onPrev: () => void;
  onNext: () => void;
};

const StepActions: React.FC<StepActionsProps> = ({
  activeStep,
  isBusy,
  canProceedTo,
  onPrev,
  onNext,
}) => {
  const isMobile = useIsMobile();

  if (isMobile) {
    return (
      <MobileBottomBar
        activeStep={activeStep}
        isBusy={isBusy}
        canProceedTo={canProceedTo}
        onPrev={onPrev}
        onNext={onNext}
      />
    );
  }

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex gap-4">
      {activeStep > 1 && (
        <button
          type="button"
          onClick={onPrev}
          disabled={isBusy}
          className={`bg-white border-2 border-slate-100 text-slate-600 font-bold px-6 py-3 rounded-2xl shadow-xl hover:bg-slate-50 transition-all flex items-center gap-2 ${
            isBusy ? 'opacity-60 cursor-not-allowed' : ''
          }`}
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7" />
          </svg>
          Previous
        </button>
      )}
      {canProceedTo(activeStep + 1) && activeStep < 4 && (
        <button
          type="button"
          onClick={onNext}
          disabled={isBusy}
          className={`bg-teal-600 text-white font-bold px-8 py-3 rounded-2xl shadow-xl shadow-teal-600/20 hover:bg-teal-700 transition-all flex items-center gap-2 ${
            isBusy ? 'opacity-60 cursor-not-allowed' : 'animate-bounce-short'
          }`}
        >
          Next Stage
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7" />
          </svg>
        </button>
      )}
    </div>
  );
};

export default StepActions;
