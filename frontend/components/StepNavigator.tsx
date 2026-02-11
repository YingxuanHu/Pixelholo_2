import React from 'react';
import MobileStepNav from '../mobile/MobileStepNav';
import { useIsMobile } from '../mobile/useIsMobile';

type StepNavigatorProps = {
  activeStep: number;
  isBusy: boolean;
  canProceedTo: (step: number) => boolean;
  onStepSelect: (step: number) => void;
};

const StepNavigator: React.FC<StepNavigatorProps> = ({
  activeStep,
  isBusy,
  canProceedTo,
  onStepSelect,
}) => {
  const isMobile = useIsMobile();

  if (isMobile) {
    return (
      <MobileStepNav
        activeStep={activeStep}
        isBusy={isBusy}
        canProceedTo={canProceedTo}
        onStepSelect={onStepSelect}
      />
    );
  }

  return (
    <div className="flex items-center justify-between relative mb-12">
      <div className="absolute top-1/2 left-0 w-full h-0.5 bg-slate-100 -translate-y-1/2 -z-10"></div>
      {[1, 2, 3, 4].map((step) => (
        <button
          key={step}
          type="button"
          onClick={() => onStepSelect(step)}
          disabled={isBusy || !canProceedTo(step)}
          className={`
            relative flex items-center justify-center w-12 h-12 rounded-full border-2 font-bold transition-all duration-300
            ${
              activeStep === step
                ? 'bg-teal-600 text-white border-teal-600 scale-110 shadow-lg shadow-teal-600/20'
                : canProceedTo(step) && !isBusy
                  ? 'bg-white text-teal-600 border-teal-600 cursor-pointer'
                  : 'bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed'
            }
          `}
        >
          {step}
          <span
            className={`absolute -bottom-7 left-1/2 -translate-x-1/2 text-[10px] uppercase tracking-widest font-bold whitespace-nowrap ${
              activeStep === step ? 'text-teal-600' : 'text-slate-400'
            }`}
          >
            {step === 1 ? 'Profile' : step === 2 ? 'Preprocess' : step === 3 ? 'Training' : 'Generation'}
          </span>
        </button>
      ))}
    </div>
  );
};

export default StepNavigator;
