# Few-Shot Extractor Evaluation Report
Generated: 2025-08-20 21:41:12

## Summary Statistics

- **Total Tests**: 3
- **Success Rate**: 100.0%
- **Schema Validity Rate**: 100.0%

### Performance Metrics
- **Average Extraction Time**: 6.800s
- **Min/Max Time**: 2.490s / 14.686s

- **Average Confidence**: 1.000
- **Min/Max Confidence**: 1.000 / 1.000

### Accuracy Metrics

- **Field Accuracy**: 55.6% (min: 33.3%, max: 100.0%)
- **Classification Accuracy**: 100.0% (min: 100.0%, max: 100.0%)
- **Extraction Accuracy**: 55.6% (min: 33.3%, max: 100.0%)

## Extractor Configuration

- **Examples Count**: 1
- **Schema Fields**: 3
- **Example Sources**: contract1

### Hybrid Fields (Extract + Classify)

- **payment_terms**: ['monthly', 'yearly', 'one-time']
- **warranty**: ['standard', 'non_standard']

### Pure Extraction Fields

- **customer_name**: The name of the customer for the contract.

## Individual Test Results

### Contract1_RealData

- **Extraction Time**: 14.686s
- **Confidence**: 1.000
- **Success**: ✅
- **Schema Valid**: ✅
- **Field Accuracy**: 100.0%
- **Classification Accuracy**: 100.0%
- **Extraction Accuracy**: 100.0%

### Contract2_Synthetic

- **Extraction Time**: 3.226s
- **Confidence**: 1.000
- **Success**: ✅
- **Schema Valid**: ✅
- **Field Accuracy**: 33.3%
- **Classification Accuracy**: 100.0%
- **Extraction Accuracy**: 33.3%

### Contract3_Minimal

- **Extraction Time**: 2.490s
- **Confidence**: 1.000
- **Success**: ✅
- **Schema Valid**: ✅
- **Field Accuracy**: 33.3%
- **Classification Accuracy**: 100.0%
- **Extraction Accuracy**: 33.3%
