from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.dtk import DTK
from algorithms.spda import SPDA
from algorithms.madi import MaDi

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
	"madi": MaDi,
	'dtk': DTK,
	'spda': SPDA,
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)

def make_agent_madi(obs_shape, action_shape, args, agent_args: dict):
	return algorithm[args.algorithm](obs_shape, action_shape, args, **agent_args)
