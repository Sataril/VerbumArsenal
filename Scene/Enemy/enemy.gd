extends CharacterBody2D
class_name Enemy

@onready var chase_area: Area2D = $ChaseArea
@onready var animated_sprite_2d: AnimatedSprite2D = $AnimatedSprite2D
@onready var attack_area: Area2D = $AttackArea
@onready var health: Health = $Health


@export var knock_back_force : float = 250
@export var move_speed: int = 50
@export var attack_cooldown: bool = false
# 新增死亡状态标识
var is_dead: bool = false
var is_attacking: bool = false

var player: Player
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	chase_area.body_entered.connect(on_body_enter_chase_area)
	chase_area.body_exited.connect(on_body_exit_chase_area)
	attack_area.body_entered.connect(on_body_enter_attack_area)
	
	animated_sprite_2d.animation_finished.connect(_on_animation_finished)
	
	health.died.connect(on_died)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	chase_player()
	

func on_body_enter_chase_area(body: Node2D)-> void:
	if body is Player:
		player = body
		
func on_body_exit_chase_area(body: Node2D)-> void:
	if body is Player:
		player = null
		
func on_body_enter_attack_area(body: Node2D)-> void:
	if body is Player:
		var player_health: Health = body.get_node_or_null("Health")
		if player_health != null:
			var damage_amount = 40
			player_health.damage(damage_amount)
			
		var knock_back: KnockBack = body.get_node_or_null("KnockBack")
		if knock_back != null:
			var direction: Vector2 = (body.global_position - global_position).normalized()
			
			var duration: float = 0.1
			knock_back.apply_knock_back(direction, knock_back_force, duration)
		
		if body is Player and not is_attacking and not attack_cooldown:
			start_attack()
			
# 统一攻击启动方法
func start_attack() -> void:
	if is_dead or attack_cooldown:
		return
	
	is_attacking = true
	attack_cooldown = true
	animated_sprite_2d.play("attack")
	velocity = Vector2.ZERO  # 停止移动	

# 攻击完成处理
func handle_attack_completion() -> void:
	is_attacking = false
	
	# 启动冷却计时器
	await get_tree().create_timer(0.5).timeout
	attack_cooldown = false
	
	# 如果玩家仍在追击范围则继续追击
	if player != null:
		animated_sprite_2d.play("run")

func chase_player()-> void:
	if is_attacking or is_dead:
		return
	
	var chase_direction: Vector2 = Vector2.ZERO
	if player != null:
		chase_direction = (player.global_position - global_position).normalized()
	velocity = chase_direction * move_speed
	
	handle_animation(chase_direction)
	handle_rotation(chase_direction)
	move_and_slide()
	
	
func handle_animation(move_direction: Vector2) -> void:
	if is_attacking or is_dead:
		return
		
	if move_direction.length() > 0:
		animated_sprite_2d.play("run")
	else:
		animated_sprite_2d.play("idle")

func handle_rotation(move_direction: Vector2) -> void:
	if move_direction.x > 0:
		animated_sprite_2d.scale = Vector2(1,1)
	elif move_direction.x < 0:
		animated_sprite_2d.scale = Vector2(-1,1)
		
# 的动画结束处理
func _on_animation_finished() -> void:
	match animated_sprite_2d.animation:
		"attack":
			handle_attack_completion()
		"death":
			queue_free()

func on_died() -> void:
	
	if is_dead:  # 防止重复触发
		return
	
	is_dead = true
	# 停止所有行为
	player = null  # 停止追踪玩家
	set_process(false)  # 禁用 _process 逻辑
	chase_area.monitoring = false  # 关闭检测区域
	attack_area.monitoring = false  # 关闭攻击区域
	
	# 播放死亡动画
	animated_sprite_2d.play("death")
	# 连接动画结束信号
	animated_sprite_2d.animation_finished.connect(_on_death_animation_finished)

# 动画结束回调
func _on_death_animation_finished() -> void:
	if animated_sprite_2d.animation == "death":
		queue_free()  # 动画播放完毕后销毁
