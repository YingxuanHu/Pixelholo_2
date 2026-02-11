import React from 'react';

type MobileStepNavProps = {
  activeStep: number;
  isBusy: boolean;
  canProceedTo: (step: number) => boolean;
  onStepSelect: (step: number) => void;
};

const MOBILE_STEPS = [
  { step: 1, label: 'Profile' },
  { step: 2, label: 'Prep' },
  { step: 3, label: 'Train' },
  { step: 4, label: 'Generate' },
];

const MobileStepNav: React.FC<MobileStepNavProps> = ({ activeStep, isBusy, canProceedTo, onStepSelect }) => {
  return (
    <div className="md:hidden mb-6">
      <div className="rounded-2xl border border-slate-800 bg-slate-950/95 backdrop-blur px-3 py-3 shadow-xl shadow-slate-900/30">
        <div className="grid grid-cols-4 gap-2">
          {MOBILE_STEPS.map(({ step, label }) => {
            const enabled = canProceedTo(step) && !isBusy;
            const active = activeStep === step;
            return (
              <button
                key={step}
                type="button"
                onClick={() => onStepSelect(step)}
                disabled={!enabled}
                className={`rounded-xl px-1 py-2 text-center transition-all border ${
                  active
                    ? 'bg-teal-500/20 border-teal-400 text-teal-200 shadow-lg shadow-teal-500/20'
                    : enabled
                    ? 'bg-slate-900 border-slate-700 text-slate-300'
                    : 'bg-slate-900/60 border-slate-800 text-slate-600'
                }`}
              >
                <div className="text-[10px] font-black tracking-widest leading-none">{step}</div>
                <div className="mt-1 text-[10px] font-semibold truncate">{label}</div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default MobileStepNav;
