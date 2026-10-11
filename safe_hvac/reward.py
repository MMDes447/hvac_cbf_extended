"""A linear comfort/energy reward using the same physical bounds as safety labels."""
from .safety import SafetySpec


class PhysicalComfortReward:
    def __init__(self, safety, energy_weight=.25, lambda_energy=1e-5, lambda_temperature=1.):
        self.safety = SafetySpec(**safety)
        self.energy_weight = energy_weight
        self.lambda_energy, self.lambda_temperature = lambda_energy, lambda_temperature

    def __call__(self, observations):
        assessment = self.safety.assess(observations)
        power = assessment["power_w"]
        violation = float(assessment["deviation_c"].sum())
        energy = -self.energy_weight * self.lambda_energy * power
        comfort = -(1 - self.energy_weight) * self.lambda_temperature * violation
        return energy + comfort, dict(energy_term=energy, comfort_term=comfort,
                                      energy_penalty=-power, comfort_penalty=-violation,
                                      total_power_demand=power, total_temperature_violation=violation,
                                      reward_weight=self.energy_weight)
