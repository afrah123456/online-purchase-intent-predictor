from typing import Dict, List, Tuple
from datetime import datetime
import time


class RiskScorer:
    """Handles real-time customer risk scoring"""

    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.risk_thresholds = {
            'critical': 0.80,  # >80% likely to abandon
            'high': 0.60,
            'medium': 0.40,
            'low': 0.20
        }

    def calculate_risk_score(self, prediction: int, probability: float) -> Dict:
        """
        Calculate comprehensive risk score

        Args:
            prediction: 0 (abandon) or 1 (purchase)
            probability: Model probability

        Returns:
            Risk assessment dictionary
        """
        # If prediction is 0 (abandon), risk is high
        # If prediction is 1 (purchase), risk is low
        if prediction == 0:
            risk_score = probability  # High probability of abandoning = high risk
        else:
            risk_score = 1 - probability  # High probability of purchase = low risk

        # Determine risk level
        if risk_score >= self.risk_thresholds['critical']:
            risk_level = 'CRITICAL'
            priority = 1
        elif risk_score >= self.risk_thresholds['high']:
            risk_level = 'HIGH'
            priority = 2
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = 'MEDIUM'
            priority = 3
        else:
            risk_level = 'LOW'
            priority = 4

        return {
            'risk_score': round(risk_score, 4),
            'risk_level': risk_level,
            'priority': priority,
            'will_convert': prediction == 1,
            'conversion_probability': round(probability, 4) if prediction == 1 else round(1 - probability, 4)
        }

    def get_intervention_actions(self, risk_score: float, input_data: Dict) -> List[Dict]:
        """
        Suggest specific actions based on risk level and customer behavior

        Returns list of actionable recommendations with expected impact
        """
        actions = []

        # Critical risk - immediate intervention needed
        if risk_score >= 0.80:
            actions.append({
                'action': 'Trigger exit-intent popup with 15% discount',
                'timing': 'immediate',
                'expected_impact': '+18% conversion',
                'priority': 'critical'
            })
            actions.append({
                'action': 'Activate live chat with proactive greeting',
                'timing': 'within 10 seconds',
                'expected_impact': '+12% conversion',
                'priority': 'critical'
            })
            actions.append({
                'action': 'Show free shipping banner if cart > $50',
                'timing': 'immediate',
                'expected_impact': '+8% conversion',
                'priority': 'high'
            })

        # High risk - strong intervention
        elif risk_score >= 0.60:
            actions.append({
                'action': 'Display limited-time offer countdown',
                'timing': 'within 20 seconds',
                'expected_impact': '+10% conversion',
                'priority': 'high'
            })
            actions.append({
                'action': 'Show customer reviews and social proof',
                'timing': 'immediate',
                'expected_impact': '+7% conversion',
                'priority': 'high'
            })
            actions.append({
                'action': 'Enable live chat option',
                'timing': 'within 30 seconds',
                'expected_impact': '+5% conversion',
                'priority': 'medium'
            })

        # Medium risk - gentle nudges
        elif risk_score >= 0.40:
            actions.append({
                'action': 'Highlight best-selling products',
                'timing': 'within 1 minute',
                'expected_impact': '+5% conversion',
                'priority': 'medium'
            })
            actions.append({
                'action': 'Show "Customers also bought" recommendations',
                'timing': 'immediate',
                'expected_impact': '+4% conversion',
                'priority': 'medium'
            })

        # Low risk - maintain engagement
        else:
            actions.append({
                'action': 'Show personalized product recommendations',
                'timing': 'within 2 minutes',
                'expected_impact': '+3% conversion',
                'priority': 'low'
            })
            actions.append({
                'action': 'Offer loyalty points for purchase',
                'timing': 'at checkout',
                'expected_impact': '+2% conversion',
                'priority': 'low'
            })

        # Specific behavior-based actions
        if input_data.get('BounceRates', 0) > 0.15:
            actions.append({
                'action': 'Improve page load speed - high bounce detected',
                'timing': 'immediate technical fix',
                'expected_impact': '+15% conversion',
                'priority': 'critical'
            })

        if input_data.get('PageValues', 0) == 0 and input_data.get('ProductRelated', 0) > 5:
            actions.append({
                'action': 'Check tracking setup - viewing products but no value tracked',
                'timing': 'investigate immediately',
                'expected_impact': 'data quality issue',
                'priority': 'high'
            })

        return actions

    def calculate_time_to_decision(self, input_data: Dict) -> str:
        """
        Estimate how long until customer makes purchase/abandon decision
        Based on browsing behavior patterns
        """
        product_pages = input_data.get('ProductRelated', 0)
        bounce_rate = input_data.get('BounceRates', 0)

        # More product pages = taking time to decide
        # High bounce = quick decision (abandon)

        if bounce_rate > 0.2:
            return "~15-30 seconds (high bounce risk)"
        elif product_pages > 15:
            return "~2-3 minutes (deep browsing)"
        elif product_pages > 5:
            return "~45-90 seconds (comparing options)"
        else:
            return "~20-40 seconds (quick browse)"

    def get_confidence_level(self, probability: float, input_data: Dict) -> Tuple[str, float]:
        """
        Calculate prediction confidence with reliability score

        Returns (confidence_label, confidence_score)
        """
        # Base confidence on probability extremes
        if probability < 0.2 or probability > 0.8:
            base_confidence = 0.9
        elif probability < 0.3 or probability > 0.7:
            base_confidence = 0.75
        elif probability < 0.4 or probability > 0.6:
            base_confidence = 0.6
        else:
            base_confidence = 0.45

        # Adjust based on data quality
        if input_data.get('PageValues', 0) == 0 and input_data.get('ProductRelated', 0) > 0:
            base_confidence *= 0.85  # Potential tracking issue

        if input_data.get('BounceRates', 0) > 0.3:
            base_confidence *= 0.9  # Very high bounce is reliable signal

        # Convert to label
        if base_confidence >= 0.8:
            label = "very_high"
        elif base_confidence >= 0.65:
            label = "high"
        elif base_confidence >= 0.5:
            label = "medium"
        else:
            label = "low"

        return label, round(base_confidence, 3)

    def generate_risk_report(self, session_data: Dict, prediction: int,
                             probability: float, model_name: str,
                             response_time_ms: float) -> Dict:
        """
        Generate comprehensive risk scoring report

        Args:
            session_data: Customer session input
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            model_name: Model used
            response_time_ms: API response time

        Returns:
            Complete risk assessment report
        """
        start_time = time.time()

        # Calculate risk score
        risk_info = self.calculate_risk_score(prediction, probability)

        # Get intervention actions
        actions = self.get_intervention_actions(risk_info['risk_score'], session_data)

        # Estimate time to decision
        time_to_decision = self.calculate_time_to_decision(session_data)

        # Get confidence level
        confidence_label, confidence_score = self.get_confidence_level(probability, session_data)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        report = {
            'session_id': session_data.get('session_id', f"session_{int(time.time())}"),
            'timestamp': datetime.utcnow().isoformat(),

            # Risk Assessment
            'risk_assessment': {
                'risk_score': risk_info['risk_score'],
                'risk_level': risk_info['risk_level'],
                'priority': risk_info['priority'],
                'will_convert': risk_info['will_convert'],
                'conversion_probability': risk_info['conversion_probability']
            },

            # Predictions
            'prediction': {
                'outcome': 'PURCHASE' if prediction == 1 else 'ABANDON',
                'probability': round(probability, 4),
                'model_used': model_name,
                'confidence': confidence_label,
                'confidence_score': confidence_score
            },

            # Recommended Actions
            'recommended_actions': actions,

            # Timing
            'timing': {
                'time_to_decision': time_to_decision,
                'optimal_intervention_window': 'next 30-60 seconds' if risk_info[
                                                                           'risk_score'] > 0.6 else 'within 2 minutes'
            },

            # Performance
            'performance': {
                'api_response_time_ms': round(response_time_ms, 2),
                'processing_time_ms': round(processing_time, 2),
                'total_time_ms': round(response_time_ms + processing_time, 2)
            },

            # Behavioral Insights
            'insights': self._generate_insights(session_data, risk_info)
        }

        return report

    def _generate_insights(self, session_data: Dict, risk_info: Dict) -> List[str]:
        """Generate human-readable insights about customer behavior"""
        insights = []

        product_pages = session_data.get('ProductRelated', 0)
        page_value = session_data.get('PageValues', 0)
        bounce_rate = session_data.get('BounceRates', 0)
        visitor_type = session_data.get('VisitorType', '')

        # Product engagement
        if product_pages > 20:
            insights.append("High product engagement - customer is seriously considering purchase")
        elif product_pages < 3:
            insights.append("Low product exploration - may need more compelling content")

        # Page value
        if page_value > 30:
            insights.append("Viewing high-value pages - strong purchase intent")
        elif page_value == 0 and product_pages > 0:
            insights.append("⚠️ Tracking issue: Viewing products but no page value recorded")

        # Bounce rate
        if bounce_rate > 0.2:
            insights.append("⚠️ High bounce rate - potential UX or performance issues")
        elif bounce_rate < 0.05:
            insights.append("Excellent engagement - low bounce rate")

        # Visitor type
        if visitor_type == 'Returning_Visitor':
            insights.append("Returning visitor - higher conversion likelihood")
        elif visitor_type == 'New_Visitor':
            insights.append("New visitor - may need more trust signals")

        # Risk-specific
        if risk_info['risk_score'] > 0.8:
            insights.append("🚨 URGENT: Very high abandonment risk - immediate action required")
        elif risk_info['risk_score'] < 0.2:
            insights.append("✅ Low risk customer - likely to convert naturally")

        return insights