from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------

register(
    id='Copy-v0',
    entry_point='gym.envs.algorithmic:CopyEnv',
    max_episode_steps=200,
    reward_threshold=25.0, )

register(
    id='RepeatCopy-v0',
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,
    reward_threshold=75.0, )

register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows': 2},
    max_episode_steps=200,
    reward_threshold=25.0, )

register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows': 3},
    max_episode_steps=200,
    reward_threshold=25.0, )

register(
    id='DuplicatedInput-v0',
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv',
    max_episode_steps=200,
    reward_threshold=9.0, )

register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0, )

# Classic
# ----------------------------------------

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0, )

register(
    id='CartPole-v1',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0, )

register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0, )

register(
    id='MountainCarContinuous-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0, )

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200, )

register(
    id='Acrobot-v1',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    max_episode_steps=500, )

# Box2d
# ----------------------------------------

register(
    id='LunarLander-v2',
    entry_point='gym.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200, )

register(
    id='LunarLanderContinuous-v2',
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200, )

register(
    id='BipedalWalker-v2',
    entry_point='gym.envs.box2d:BipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300, )

register(
    id='BipedalWalkerHardcore-v2',
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,
    reward_threshold=300, )

register(
    id='CarRacing-v0',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900, )

# Toy Text
# ----------------------------------------

register(
    id='Blackjack-v0',
    entry_point='gym.envs.toy_text:BlackjackEnv', )

register(
    id='KellyCoinflip-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipEnv',
    reward_threshold=246.61, )

register(
    id='FrozenLake-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4'},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

register(
    id='FrozenLake8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99,  # optimum = 1
)

register(
    id='NChain-v0',
    entry_point='gym.envs.toy_text:NChainEnv',
    max_episode_steps=1000, )

register(
    id='Roulette-v0',
    entry_point='gym.envs.toy_text:RouletteEnv',
    max_episode_steps=100, )

register(
    id='Taxi-v2',
    entry_point='gym.envs.toy_text.taxi:TaxiEnv',
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=200, )

register(
    id='GuessingGame-v0',
    entry_point='gym.envs.toy_text.guessing_game:GuessingGame',
    max_episode_steps=200, )

register(
    id='HotterColder-v0',
    entry_point='gym.envs.toy_text.hotter_colder:HotterColder',
    max_episode_steps=200, )

# Mujoco
# ----------------------------------------

# 2D

register(
    id='Reacher-v1',
    entry_point='gym.envs.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75, )

register(
    id='InvertedPendulum-v1',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0, )

register(
    id='InvertedDoublePendulum-v1',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0, )

register(
    id='HalfCheetah-v1',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0, )

register(
    id='Hopper-v1',
    entry_point='gym.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0, )

register(
    id='Swimmer-v1',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0, )

register(
    id='Walker2d-v1',
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco:Walker2dEnv', )

register(
    id='Ant-v1',
    entry_point='gym.envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0, )

register(
    id='Humanoid-v1',
    entry_point='gym.envs.mujoco:HumanoidEnv',
    max_episode_steps=1000, )

register(
    id='HumanoidStandup-v1',
    entry_point='gym.envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000, )

# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in [
        'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids',
        'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'berzerk',
        'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
        'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
        'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite',
        'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond',
        'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
        'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix',
        'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid',
        'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris',
        'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham',
        'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor',
        'yars_revenge', 'zaxxon'
]:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={
                'game': game,
                'obs_type': obs_type,
                'repeat_action_probability': 0.25
            },
            max_episode_steps=10000,
            nondeterministic=nondeterministic, )

        register(
            id='{}-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game,
                    'obs_type': obs_type},
            max_episode_steps=100000,
            nondeterministic=nondeterministic, )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={
                'game': game,
                'obs_type': obs_type,
                'frameskip': frameskip,
                'repeat_action_probability': 0.25
            },
            max_episode_steps=100000,
            nondeterministic=nondeterministic, )

        register(
            id='{}Deterministic-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={
                'game': game,
                'obs_type': obs_type,
                'frameskip': frameskip
            },
            max_episode_steps=100000,
            nondeterministic=nondeterministic, )

        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={
                'game': game,
                'obs_type': obs_type,
                'frameskip': 1,
                'repeat_action_probability': 0.25
            },  # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic, )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v3'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game,
                    'obs_type': obs_type,
                    'frameskip':
                    1},  # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic, )

# Board games
# ----------------------------------------

register(
    id='Go9x9-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',
        'observation_type': 'image3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    },
    # The pachi player seems not to be determistic given a fixed seed.
    # (Reproduce by running 'import gym; h = gym.make('Go9x9-v0'); h.seed(1); h.reset(); h.step(15); h.step(16); h.step(17)' a few times.)
    #
    # This is probably due to a computation time limit.
    nondeterministic=True, )

register(
    id='Go19x19-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',
        'observation_type': 'image3c',
        'illegal_move_mode': 'lose',
        'board_size': 19,
    },
    nondeterministic=True, )

register(
    id='Hex9x9-v0',
    entry_point='gym.envs.board_game:HexEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    }, )

# Debugging
# ----------------------------------------

register(
    id='OneRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundDeterministicRewardEnv',
    local_only=True)

register(
    id='TwoRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundDeterministicRewardEnv',
    local_only=True)

register(
    id='OneRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundNondeterministicRewardEnv',
    local_only=True)

register(
    id='TwoRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundNondeterministicRewardEnv',
    local_only=True, )

# Parameter tuning
# ----------------------------------------
register(
    id='ConvergenceControl-v0',
    entry_point='gym.envs.parameter_tuning:ConvergenceControl', )

register(
    id='CNNClassifierTraining-v0',
    entry_point='gym.envs.parameter_tuning:CNNClassifierTraining', )

# Safety
# ----------------------------------------

# interpretability envs
register(
    id='PredictActionsCartpole-v0',
    entry_point='gym.envs.safety:PredictActionsCartpoleEnv',
    max_episode_steps=200, )

register(
    id='PredictObsCartpole-v0',
    entry_point='gym.envs.safety:PredictObsCartpoleEnv',
    max_episode_steps=200, )

# semi_supervised envs
# probably the easiest:
register(
    id='SemisuperPendulumNoise-v0',
    entry_point='gym.envs.safety:SemisuperPendulumNoiseEnv',
    max_episode_steps=200, )
# somewhat harder because of higher variance:
register(
    id='SemisuperPendulumRandom-v0',
    entry_point='gym.envs.safety:SemisuperPendulumRandomEnv',
    max_episode_steps=200, )
# probably the hardest because you only get a constant number of rewards in total:
register(
    id='SemisuperPendulumDecay-v0',
    entry_point='gym.envs.safety:SemisuperPendulumDecayEnv',
    max_episode_steps=200, )

# off_switch envs
register(
    id='OffSwitchCartpole-v0',
    entry_point='gym.envs.safety:OffSwitchCartpoleEnv',
    max_episode_steps=200, )

register(
    id='OffSwitchCartpoleProb-v0',
    entry_point='gym.envs.safety:OffSwitchCartpoleProbEnv',
    max_episode_steps=200, )

# RDDL domains

register(
    id='RDDL-sysadmin1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '1'})


register(
    id='RDDL-sysadmin2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '2'})


register(
    id='RDDL-sysadmin3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '3'})


register(
    id='RDDL-sysadmin4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '4'})


register(
    id='RDDL-sysadmin5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '5'})


register(
    id='RDDL-sysadmin6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '6'})


register(
    id='RDDL-sysadmin7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '7'})


register(
    id='RDDL-sysadmin8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '8'})


register(
    id='RDDL-sysadmin9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '9'})


register(
    id='RDDL-sysadmin10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '10'})

register(
    id='RDDL-sysadmin1.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '1_1'})

register(
    id='RDDL-sysadmin1.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '1_2'})

register(
    id='RDDL-sysadmin1.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '1_3'})

register(
    id='RDDL-sysadmin1.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '1_4'})

register(
    id='RDDL-sysadmin1.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '1_5'})

register(
    id='RDDL-sysadmin1.6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '1_6'})

register(
    id='RDDL-sysadmin5.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '5_1'})

register(
    id='RDDL-sysadmin5.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '5_2'})

register(
    id='RDDL-sysadmin5.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '5_3'})

register(
    id='RDDL-sysadmin5.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '5_4'})

register(
    id='RDDL-sysadmin5.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '5_5'})

register(
    id='RDDL-sysadmin10.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '10_1'})

register(
    id='RDDL-sysadmin10.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '10_2'})

register(
    id='RDDL-sysadmin10.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '10_3'})

register(
    id='RDDL-sysadmin10.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '10_4'})

register(
    id='RDDL-sysadmin10.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'sysadmin',
            'instance': '10_5'})

register(
    id='RDDL-gameoflife1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '1'})

register(
    id='RDDL-gameoflife2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '2'})


register(
    id='RDDL-gameoflife3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '3'})


register(
    id='RDDL-gameoflife4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '4'})


register(
    id='RDDL-gameoflife5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '5'})


register(
    id='RDDL-gameoflife6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '6'})


register(
    id='RDDL-gameoflife7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '7'})


register(
    id='RDDL-gameoflife8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '8'})


register(
    id='RDDL-gameoflife9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '9'})


register(
    id='RDDL-gameoflife10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '10'})

register(
    id='RDDL-gameoflife1.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '1_1'})

register(
    id='RDDL-gameoflife1.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '1_2'})

register(
    id='RDDL-gameoflife1.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '1_3'})

register(
    id='RDDL-gameoflife1.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '1_4'})

register(
    id='RDDL-gameoflife1.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '1_5'})

register(
    id='RDDL-gameoflife5.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '5_1'})

register(
    id='RDDL-gameoflife5.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '5_2'})

register(
    id='RDDL-gameoflife5.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '5_3'})

register(
    id='RDDL-gameoflife5.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '5_4'})

register(
    id='RDDL-gameoflife10.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '10_1'})

register(
    id='RDDL-gameoflife10.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '10_2'})

register(
    id='RDDL-gameoflife10.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '10_3'})

register(
    id='RDDL-gameoflife10.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '10_4'})

register(
    id='RDDL-gameoflife10.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'gameoflife',
            'instance': '10_5'})

register(
    id='RDDL-navigation1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '1'})


register(
    id='RDDL-navigation2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '2'})


register(
    id='RDDL-navigation3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '3'})


register(
    id='RDDL-navigation4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '4'})


register(
    id='RDDL-navigation5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '5'})


register(
    id='RDDL-navigation6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '6'})


register(
    id='RDDL-navigation7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '7'})


register(
    id='RDDL-navigation8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '8'})


register(
    id='RDDL-navigation9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '9'})


register(
    id='RDDL-navigation10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '10'})

register(
    id='RDDL-navigation1.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '1_1'})

register(
    id='RDDL-navigation1.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '1_2'})

register(
    id='RDDL-navigation1.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '1_3'})

register(
    id='RDDL-navigation1.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '1_4'})

register(
    id='RDDL-navigation1.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '1_5'})

register(
    id='RDDL-navigation2.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '2_1'})

register(
    id='RDDL-navigation2.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '2_2'})

register(
    id='RDDL-navigation2.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '2_3'})

register(
    id='RDDL-navigation2.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '2_4'})

register(
    id='RDDL-navigation2.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '2_5'})

register(
    id='RDDL-navigation3.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '3_1'})

register(
    id='RDDL-navigation3.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '3_2'})

register(
    id='RDDL-navigation3.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '3_3'})

register(
    id='RDDL-navigation3.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '3_4'})

register(
    id='RDDL-navigation3.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'navigation',
            'instance': '3_5'})

register(
    id='RDDL-tamarisk1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '1'})


register(
    id='RDDL-tamarisk2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '2'})


register(
    id='RDDL-tamarisk3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '3'})


register(
    id='RDDL-tamarisk4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '4'})


register(
    id='RDDL-tamarisk5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '5'})


register(
    id='RDDL-tamarisk6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '6'})


register(
    id='RDDL-tamarisk7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '7'})


register(
    id='RDDL-tamarisk8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '8'})


register(
    id='RDDL-tamarisk9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '9'})


register(
    id='RDDL-tamarisk10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'tamarisk',
            'instance': '10'})


register(
    id='RDDL-elevators1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '1'})


register(
    id='RDDL-elevators2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '2'})


register(
    id='RDDL-elevators3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '3'})


register(
    id='RDDL-elevators4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '4'})


register(
    id='RDDL-elevators5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '5'})


register(
    id='RDDL-elevators6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '6'})


register(
    id='RDDL-elevators7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '7'})


register(
    id='RDDL-elevators8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '8'})


register(
    id='RDDL-elevators9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '9'})


register(
    id='RDDL-elevators10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'elevators',
            'instance': '10'})


register(
    id='RDDL-traffic1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '1'})


register(
    id='RDDL-traffic2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '2'})


register(
    id='RDDL-traffic3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '3'})


register(
    id='RDDL-traffic4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '4'})


register(
    id='RDDL-traffic5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '5'})


register(
    id='RDDL-traffic6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '6'})


register(
    id='RDDL-traffic7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '7'})


register(
    id='RDDL-traffic8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '8'})


register(
    id='RDDL-traffic9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '9'})


register(
    id='RDDL-traffic10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'traffic',
            'instance': '10'})


register(
    id='RDDL-skill_teaching1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '1'})


register(
    id='RDDL-skill_teaching2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '2'})


register(
    id='RDDL-skill_teaching3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '3'})


register(
    id='RDDL-skill_teaching4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '4'})


register(
    id='RDDL-skill_teaching5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '5'})


register(
    id='RDDL-skill_teaching6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '6'})


register(
    id='RDDL-skill_teaching7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '7'})


register(
    id='RDDL-skill_teaching8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '8'})


register(
    id='RDDL-skill_teaching9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '9'})


register(
    id='RDDL-skill_teaching10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'skill_teaching',
            'instance': '10'})


register(
    id='RDDL-recon1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '1'})


register(
    id='RDDL-recon2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '2'})


register(
    id='RDDL-recon3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '3'})


register(
    id='RDDL-recon4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '4'})


register(
    id='RDDL-recon5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '5'})


register(
    id='RDDL-recon6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '6'})


register(
    id='RDDL-recon7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '7'})


register(
    id='RDDL-recon8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '8'})


register(
    id='RDDL-recon9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '9'})


register(
    id='RDDL-recon10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'recon',
            'instance': '10'})


register(
    id='RDDL-academic_advising1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '1'})


register(
    id='RDDL-academic_advising2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '2'})


register(
    id='RDDL-academic_advising3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '3'})


register(
    id='RDDL-academic_advising4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '4'})


register(
    id='RDDL-academic_advising5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '5'})


register(
    id='RDDL-academic_advising6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '6'})


register(
    id='RDDL-academic_advising7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '7'})


register(
    id='RDDL-academic_advising8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '8'})


register(
    id='RDDL-academic_advising9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '9'})


register(
    id='RDDL-academic_advising10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'academic_advising',
            'instance': '10'})


register(
    id='RDDL-crossing_traffic1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '1'})


register(
    id='RDDL-crossing_traffic2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '2'})


register(
    id='RDDL-crossing_traffic3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '3'})


register(
    id='RDDL-crossing_traffic4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '4'})


register(
    id='RDDL-crossing_traffic5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '5'})


register(
    id='RDDL-crossing_traffic6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '6'})


register(
    id='RDDL-crossing_traffic7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '7'})


register(
    id='RDDL-crossing_traffic8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '8'})


register(
    id='RDDL-crossing_traffic9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '9'})


register(
    id='RDDL-crossing_traffic10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'crossing_traffic',
            'instance': '10'})


register(
    id='RDDL-triangle_tireworld1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '1'})


register(
    id='RDDL-triangle_tireworld2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '2'})


register(
    id='RDDL-triangle_tireworld3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '3'})


register(
    id='RDDL-triangle_tireworld4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '4'})


register(
    id='RDDL-triangle_tireworld5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '5'})


register(
    id='RDDL-triangle_tireworld6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '6'})


register(
    id='RDDL-triangle_tireworld7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '7'})


register(
    id='RDDL-triangle_tireworld8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '8'})


register(
    id='RDDL-triangle_tireworld9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '9'})


register(
    id='RDDL-triangle_tireworld10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'triangle_tireworld',
            'instance': '10'})


register(
    id='RDDL-wildfire1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '1'})


register(
    id='RDDL-wildfire2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '2'})


register(
    id='RDDL-wildfire3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '3'})


register(
    id='RDDL-wildfire4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '4'})


register(
    id='RDDL-wildfire5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '5'})


register(
    id='RDDL-wildfire6-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '6'})


register(
    id='RDDL-wildfire7-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '7'})


register(
    id='RDDL-wildfire8-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '8'})


register(
    id='RDDL-wildfire9-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '9'})


register(
    id='RDDL-wildfire10-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '10'})

register(
    id='RDDL-wildfire1.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '1_1'})

register(
    id='RDDL-wildfire1.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '1_2'})

register(
    id='RDDL-wildfire1.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '1_3'})

register(
    id='RDDL-wildfire1.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '1_4'})

register(
    id='RDDL-wildfire1.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '1_5'})

register(
    id='RDDL-wildfire5.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '5_1'})

register(
    id='RDDL-wildfire5.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '5_2'})

register(
    id='RDDL-wildfire5.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '5_3'})

register(
    id='RDDL-wildfire5.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '5_4'})

register(
    id='RDDL-wildfire5.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '5_5'})

register(
    id='RDDL-wildfire10.1-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '10_1'})

register(
    id='RDDL-wildfire10.2-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '10_2'})

register(
    id='RDDL-wildfire10.3-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '10_3'})

register(
    id='RDDL-wildfire10.4-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '10_4'})

register(
    id='RDDL-wildfire10.5-v1',
    entry_point='gym.envs.rddl:RDDLEnv',
    kwargs={'domain': 'wildfire',
            'instance': '10_5'})
