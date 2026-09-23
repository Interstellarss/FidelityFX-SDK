// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE
#define FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE 0 // Reference
#endif

// Quintic Lagrange preserves fractional translation without the repeated
// high-frequency attenuation of the cubic history filter. Keep FP32 weights.
void HistoryWeights(FfxFloat32 t, FFX_PARAMETER_OUT FfxFloat32 w[6])
{
    w[0] = -(t + 1.0f) * t * (t - 1.0f) * (t - 2.0f) * (t - 3.0f) / 120.0f;
    w[1] = (t + 2.0f) * t * (t - 1.0f) * (t - 2.0f) * (t - 3.0f) / 24.0f;
    w[2] = -(t + 2.0f) * (t + 1.0f) * (t - 1.0f) * (t - 2.0f) * (t - 3.0f) / 12.0f;
    w[3] = (t + 2.0f) * (t + 1.0f) * t * (t - 2.0f) * (t - 3.0f) / 12.0f;
    w[4] = -(t + 2.0f) * (t + 1.0f) * t * (t - 1.0f) * (t - 3.0f) / 24.0f;
    w[5] = (t + 2.0f) * (t + 1.0f) * t * (t - 1.0f) * (t - 2.0f) / 120.0f;
}

FfxFloat32x4 FilterHistoryRow(FfxFloat32x4 samples[6], FfxFloat32 weights[6])
{
    FfxFloat32x4 value = FfxFloat32x4(0.0f, 0.0f, 0.0f, 0.0f);
    FFX_UNROLL
    for (FfxInt32 i = 0; i < 6; ++i) value += samples[i] * weights[i];
    const FfxFloat32x4 lo = ffxMin(samples[2], samples[3]);
    const FfxFloat32x4 hi = ffxMax(samples[2], samples[3]);
    // A smooth extremum can lie between the two central samples. A strict
    // central min/max clamp shaves it off at every subpixel reprojection.
    // Bound the extension by the neighboring curvature. Monotone steps and
    // flat regions retain the original hard bound (no extra ringing).
    const FfxFloat32x4 zero = FfxFloat32x4(0.0f, 0.0f, 0.0f, 0.0f);
    const FfxFloat32x4 lower = ffxMax(zero, ffxMin(samples[1] - hi, samples[4] - hi)) * 0.125f;
    const FfxFloat32x4 upper = ffxMax(zero, ffxMin(lo - samples[1], lo - samples[4])) * 0.125f;
    return clamp(value, lo - lower, hi + upper);
}

FfxFloat32x4 HistorySample(FfxFloat32x2 uv, FfxInt32x2 size)
{
    const FfxFloat32x2 pos = uv * FfxFloat32x2(size) - FfxFloat32x2(0.5f, 0.5f);
    const FfxInt32x2 base = FfxInt32x2(floor(pos));
    const FfxFloat32x2 t = ffxFract(pos);
    FfxFloat32 wx[6], wy[6];
    HistoryWeights(t.x, wx);
    HistoryWeights(t.y, wy);
    FfxFloat32x4 rows[6];
    FFX_UNROLL
    for (FfxInt32 y = 0; y < 6; ++y) {
        FfxFloat32x4 samples[6];
        FFX_UNROLL
        for (FfxInt32 x = 0; x < 6; ++x)
            samples[x] = LoadHistory(ClampCoord(base, FfxInt32x2(x - 2, y - 2), size));
        rows[y] = FilterHistoryRow(samples, wx);
    }
    return FilterHistoryRow(rows, wy);
}

FfxFloat32x2 GetMotionVector(FfxInt32x2 iPxHrPos, FfxFloat32x2 fHrUv)
{
#if FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS
    const FfxFloat32x2 fDilatedMotionVector = LoadDilatedMotionVector(FFX_MIN16_I2(fHrUv * RenderSize()));
#else
    const FfxFloat32x2 fDilatedMotionVector = LoadInputMotionVector(iPxHrPos);
#endif

    return fDilatedMotionVector;
}

void ComputeReprojectedUVs(FFX_PARAMETER_INOUT AccumulationPassCommonParams params)
{
    params.fReprojectedHrUv = params.fHrUv + params.fMotionVector;

    params.bIsExistingSample = IsUvInside(params.fReprojectedHrUv);
}

void ReprojectHistoryColor(const AccumulationPassCommonParams params, FFX_PARAMETER_INOUT AccumulationPassData data)

{
    FfxFloat32x4 fReprojectedHistory;
    if (params.fMotionVector.x == 0.0f && params.fMotionVector.y == 0.0f &&
        UpscaleSize().x == PreviousFrameUpscaleSize().x && UpscaleSize().y == PreviousFrameUpscaleSize().y) {
        fReprojectedHistory = LoadHistory(params.iPxHrPos);
    } else {
        fReprojectedHistory = HistorySample(params.fReprojectedHrUv, PreviousFrameUpscaleSize());
    }

    data.fHistoryColor = fReprojectedHistory.rgb;
    data.fHistoryColor *= DeltaPreExposure();
    data.fHistoryColor *= Exposure();

    data.fHistoryColor = RGBToYCoCg(data.fHistoryColor);

    data.fLock = fReprojectedHistory.w;
}
