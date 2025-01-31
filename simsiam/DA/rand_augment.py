
class RandAugment:
    def __init__(self, policies):
        self.policies = policies
    def getSubpolicies(self, x):
        subpolicies = []
        for i in range(len(self.policies) * 2):
            policy = self.policies[i % len(self.policies)]
            scale = x[0]
            p = x[1]
            if not policy.need_p():
                p = 1.0
            subpolicies.append(
                self.policies[i % len(self.policies)].get_entity(scale=scale,
                                                       p=p))
        return subpolicies