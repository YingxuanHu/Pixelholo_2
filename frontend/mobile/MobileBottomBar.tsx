import React from 'react';

type MobileBottomBarProps = {
  activeStep: number;
  isBusy: boolean;
  canProceedTo: (step: number) => boolean;
  onPrev: () => void;
  onNext: () => void;
};

const MobileBottomBar: React.FC<MobileBottomBarProps> = ({
  activeStep,
  isBusy,
  canProceedTo,
  onPrev,
  onNext,
}) => {
  const showPrev = activeStep > 1;
  const showNext = canProceedTo(activeStep + 1) && activeStep < 4;

  return (
    <div className="md:hidden fixed bottom-0 inset-x-0 z-50 border-t border-slate-700 bg-gradient-to-t from-slate-950 via-slate-900 to-slate-900/95 backdrop-blur px-4 pt-3 pb-[calc(env(safe-area-inset-bottom)+0.75rem)]">
      <div className="mx-auto max-w-xl flex items-center gap-3">
        {showPrev && (
          <button
            type="button"
            onClick={onPrev}
            disabled={isBusy}
            className={`flex-1 rounded-xl px-4 py-3 text-sm font-bold border border-slate-600 text-slate-100 bg-slate-800 ${
              isBusy ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            Previous
          </button>
        )}
        {showNext && (
          <button
            type="button"
            onClick={onNext}
            disabled={isBusy}
            className={`flex-1 rounded-xl px-4 py-3 text-sm font-black bg-teal-500 text-slate-950 shadow-lg shadow-teal-500/40 ${
              isBusy ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            Next Stage
          </button>
        )}
      </div>
    </div>
  );
};

export default MobileBottomBar;
