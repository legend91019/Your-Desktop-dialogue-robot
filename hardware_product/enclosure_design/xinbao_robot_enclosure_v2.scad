// Xinbao desktop robot enclosure v2
// Robot-shaped shell inspired by FrontEnd/robot.html:
// rounded head, screen face, side ears, small body/core, and soft base.
//
// Export:
//   openscad.com -o stl/xinbao_robot_front_shell_v2.stl -D PART_ID=1 xinbao_robot_enclosure_v2.scad
//   openscad.com -o stl/xinbao_robot_back_shell_v2.stl  -D PART_ID=2 xinbao_robot_enclosure_v2.scad
//   openscad.com -o stl/xinbao_robot_base_body_v2.stl   -D PART_ID=3 xinbao_robot_enclosure_v2.scad

$fn = 72;

// 0 assembly preview, 1 front head shell, 2 back head shell, 3 base/body
PART_ID = 0;

wall = 3;

// Head envelope. This is a true 3D rounded cuboid, not a straight box.
head_w = 205;
head_d = 126;
head_h = 155;
head_bottom_z = 78;
head_r = 28;
head_corner_3d_r = 24;
front_depth = 56;
seam_y = -head_d / 2 + front_depth;

// Face screen. Selected 5 inch HDMI touchscreen:
// active area 108.8 x 65.3 mm, PCB 121.11 x 77.93 mm.
screen_active_w = 108.8;
screen_active_h = 65.3;
screen_pcb_w = 121.11;
screen_pcb_h = 77.93;
screen_window_w = 112;
screen_window_h = 68;
screen_center_z = head_bottom_z + 88;
screen_mount_w = 126;
screen_mount_h = 82;
screen_pocket_w = screen_pcb_w + 4;
screen_pocket_h = screen_pcb_h + 4;
screen_screw_d = 2.8;

// M260C ring mic from STEP: about 72.9 x 75.75 x 3.25 mm.
mic_outer_d = 82;
mic_recess_depth = 1.6;
mic_hole_r = 30;
mic_hole_d = 5;
mic_center_d = 22;
mic_center_y = -18;

// M260C main board + adapter STEP: about 90 x 75.35 x 19.4 mm.
// It is kept inside the back shell, not outside the robot.
m260c_board_w = 94;
m260c_board_h = 80;
m260c_board_t = 24;

// Orange Pi AI Pro 20T official case from provided product image.
pi_case_w = 124.2;
pi_case_d = 88.6;
pi_case_h = 36;
pi_clearance = 3;

// 40 mm round speaker, magnet about 22 mm.
speaker_d = 40;
speaker_magnet_d = 22;
speaker_pod_d = 48;
speaker_grille_hole_d = 3.0;

// Lower body and base. The body is wider near the base for the Orange Pi
// case, then narrows toward the head to look more like the frontend robot.
body_w = 160;
body_d = 116;
body_h = 72;
body_bottom_z = 16;
body_top_w = 104;
body_top_d = 82;
waist_w = 118;
waist_d = 92;
base_w = 224;
base_d = 154;
base_h = 18;

module rounded_box(size, r) {
  linear_extrude(height = size[2])
    offset(r = r)
      square([size[0] - 2 * r, size[1] - 2 * r], center = true);
}

module rounded_box_3d(size, r) {
  hull() {
    for (x = [-(size[0] / 2 - r), size[0] / 2 - r])
      for (y = [-(size[1] / 2 - r), size[1] / 2 - r])
        for (z = [r, size[2] - r])
          translate([x, y, z])
            sphere(r = r, $fn = 36);
  }
}

module tapered_rounded_body(bottom_size, top_size, h, r) {
  hull() {
    translate([0, 0, r])
      rounded_box([bottom_size[0], bottom_size[1], 1], r);
    translate([0, 0, h - r])
      rounded_box([top_size[0], top_size[1], 1], min(r, min(top_size[0], top_size[1]) / 2 - 1));
  }
}

module y_cylinder(d, h) {
  rotate([90, 0, 0]) cylinder(d = d, h = h, center = true);
}

module head_outer() {
  rounded_box_3d([head_w, head_d, head_h], head_corner_3d_r);
}

module head_inner() {
  translate([0, 0, wall])
    rounded_box_3d(
      [head_w - 2 * wall, head_d - 2 * wall, head_h - wall + 1],
      head_corner_3d_r - wall
    );
}

module ear_outer(side) {
  x = side * (head_w / 2 + 8);
  y = -head_d / 2 + 20;
  zc = head_bottom_z + head_h * 0.56;
  hull() {
    translate([x, y, zc - 34]) y_cylinder(d = speaker_pod_d, h = 46);
    translate([x, y, zc + 34]) y_cylinder(d = speaker_pod_d, h = 46);
  }
}

module ears_outer() {
  ear_outer(-1);
  ear_outer(1);
}

module full_head_outer() {
  translate([0, 0, head_bottom_z]) head_outer();
  ears_outer();
}

module full_head_inner() {
  translate([0, 0, head_bottom_z]) head_inner();
  for (side = [-1, 1]) {
    x = side * (head_w / 2 + 8);
    y = -head_d / 2 + 20;
    zc = head_bottom_z + head_h * 0.56;
    hull() {
      translate([x, y, zc - 34]) y_cylinder(d = speaker_pod_d - 2 * wall, h = 46 - 2 * wall);
      translate([x, y, zc + 34]) y_cylinder(d = speaker_pod_d - 2 * wall, h = 46 - 2 * wall);
    }
  }
}

module head_shell() {
  difference() {
    full_head_outer();
    full_head_inner();
  }
}

module front_clip() {
  translate([-180, -head_d / 2 - 20, head_bottom_z - 6])
    cube([360, front_depth + 26, head_h + 28]);
}

module back_clip() {
  translate([-180, seam_y - 2, head_bottom_z - 6])
    cube([360, head_d + 46, head_h + 28]);
}

module screen_bosses() {
  y = -head_d / 2 + wall + 6;
  for (x = [-screen_pocket_w / 2, screen_pocket_w / 2])
    for (z = [screen_center_z - screen_pocket_h / 2, screen_center_z + screen_pocket_h / 2])
      translate([x, y, z])
        y_cylinder(d = 9, h = 10);
}

module face_bezel() {
  // Raised face frame to make the front read as a robot face, not a panel cut.
  difference() {
    translate([0, -head_d / 2 - 3.5, screen_center_z - screen_window_h / 2 - 10])
      rounded_box([screen_window_w + 24, 7, screen_window_h + 20], 12);
    translate([-screen_window_w / 2, -head_d / 2 - 7, screen_center_z - screen_window_h / 2])
      cube([screen_window_w, 16, screen_window_h]);
  }
}

module screen_retainer_lips() {
  // Screw hole spacing is not confirmed, so the first prototype uses a
  // loose pocket plus small retainer lips instead of relying only on holes.
  y = -head_d / 2 + wall + 9;
  lip_w = 8;
  lip_h = 7;
  for (x = [-(screen_pocket_w / 2 + lip_w / 2), screen_pocket_w / 2 + lip_w / 2])
    translate([x, y, screen_center_z])
      cube([lip_w, 8, screen_pocket_h + 2], center = true);
  for (z = [screen_center_z - screen_pocket_h / 2 - lip_h / 2, screen_center_z + screen_pocket_h / 2 + lip_h / 2])
    translate([0, y, z])
      cube([screen_pocket_w + 18, 8, lip_h], center = true);
}

module screen_screw_holes() {
  y = -head_d / 2 + wall + 6;
  for (x = [-screen_mount_w / 2, screen_mount_w / 2])
    for (z = [screen_center_z - screen_mount_h / 2, screen_center_z + screen_mount_h / 2])
      translate([x, y, z])
        y_cylinder(d = screen_screw_d, h = 24);
}

module face_cutouts() {
  // Physical screen window.
  translate([-screen_window_w / 2, -head_d / 2 - 6, screen_center_z - screen_window_h / 2])
    cube([screen_window_w, wall + 14, screen_window_h]);

  // Soft cyber LED slot above the face, matching the frontend robot.
  translate([-38, -head_d / 2 - 6, screen_center_z + 52])
    cube([76, wall + 14, 5]);

  // Small cheek accents.
  for (x = [-56, 56])
    translate([x, -head_d / 2 - 6, screen_center_z - 36])
      y_cylinder(d = 14, h = wall + 14);

  screen_screw_holes();
}

module mic_top_cutouts() {
  top_z = head_bottom_z + head_h;
  translate([0, mic_center_y, top_z - mic_recess_depth])
    cylinder(d = mic_outer_d, h = mic_recess_depth + 2);
  translate([0, mic_center_y, top_z - wall - 2])
    cylinder(d = mic_center_d, h = wall + 6);
  for (a = [0 : 60 : 300])
    translate([mic_hole_r * cos(a), mic_center_y + mic_hole_r * sin(a), top_z - wall - 2])
      cylinder(d = mic_hole_d, h = wall + 6);
}

module speaker_grille(side) {
  x0 = side * (head_w / 2 + 8);
  y = -head_d / 2 - 6;
  z0 = head_bottom_z + head_h * 0.56;
  for (x = [-15 : 6 : 15])
    for (z = [-15 : 6 : 15])
      if (sqrt(x * x + z * z) < speaker_d / 2 - 2)
        translate([x0 + x, y, z0 + z])
          y_cylinder(d = speaker_grille_hole_d, h = wall + 18);
}

module speaker_support(side) {
  x0 = side * (head_w / 2 + 8);
  y = -head_d / 2 + wall + 9;
  z0 = head_bottom_z + head_h * 0.56;
  difference() {
    translate([x0, y, z0]) y_cylinder(d = speaker_d + 8, h = 7);
    translate([x0, y, z0]) y_cylinder(d = speaker_d + 1.2, h = 9);
  }
}

module side_and_back_vents() {
  // Hidden side vents, less boxy than the v1 straight front grille.
  for (side = [-1, 1])
    for (z = [head_bottom_z + 58 : 10 : head_bottom_z + 122])
      translate([side * (head_w / 2 - 1), 20, z])
        cube([14, 48, 3.0], center = true);

  // Back cooling holes.
  for (x = [-32 : 8 : 32])
    for (z = [head_bottom_z + 52 : 8 : head_bottom_z + 108])
      if (abs(x) + abs(z - (head_bottom_z + 80)) < 68)
        translate([x, head_d / 2 + 6, z])
          y_cylinder(d = 3.2, h = wall + 16);
}

module back_service_cutouts() {
  // Service window for cables and the M260C adapter board.
  translate([-62, head_d / 2 - wall - 3, head_bottom_z + 32])
    cube([124, wall + 12, 54]);
  side_and_back_vents();
}

module m260c_board_rails() {
  y = head_d / 2 - wall - 7;
  zc = head_bottom_z + 85;
  // Clip rails for the 90 x 75 x 19.4 mm board/adapter assembly.
  translate([0, y, zc - m260c_board_h / 2 - 4])
    cube([m260c_board_w + 10, 8, 5], center = true);
  translate([0, y, zc + m260c_board_h / 2 + 4])
    cube([m260c_board_w + 10, 8, 5], center = true);
  for (x = [-(m260c_board_w / 2 + 4), m260c_board_w / 2 + 4])
    translate([x, y, zc])
      cube([5, 8, m260c_board_h + 10], center = true);
}

module front_head_shell() {
  difference() {
    union() {
      intersection() {
        head_shell();
        front_clip();
      }
      face_bezel();
      screen_bosses();
      screen_retainer_lips();
      // Use ears as speaker pods. The right pod can be empty if only one
      // speaker is installed, but the outside stays symmetrical.
      speaker_support(-1);
      speaker_support(1);
    }
    face_cutouts();
    mic_top_cutouts();
    speaker_grille(-1);
    speaker_grille(1);
  }
}

module back_head_shell() {
  difference() {
    union() {
      intersection() {
        head_shell();
        back_clip();
      }
      m260c_board_rails();
    }
    back_service_cutouts();
    mic_top_cutouts();
  }
}

module base_outer() {
  union() {
    // Wide soft plinth with a shallow raised "hologram" ring.
    rounded_box([base_w, base_d, base_h], 30);
    translate([0, 0, base_h - 1])
      difference() {
        rounded_box([base_w - 24, base_d - 20, 4], 26);
        translate([0, 0, -1])
          rounded_box([base_w - 52, base_d - 44, 7], 20);
      }

    // Rounded tapered torso instead of a rectangular pedestal.
    translate([0, 0, body_bottom_z])
      tapered_rounded_body([body_w, body_d], [body_top_w, body_top_d], body_h, 22);

    // Wide continuous saddle under the head. This removes the visual and
    // structural "floating head" problem while keeping a softer robot body.
    translate([0, -4, body_bottom_z + body_h - 8])
      rounded_box_3d([waist_w, waist_d, 30], 18);

    // Front cheek-like decorative panels are now embedded in the torso
    // instead of floating side cylinders.
    for (side = [-1, 1])
      translate([side * 56, -body_d / 2 - 2, body_bottom_z + 32])
        y_cylinder(d = 20, h = 6);
  }
}

module pi_case_tray() {
  tray_w = pi_case_w + 2 * pi_clearance;
  tray_d = pi_case_d + 2 * pi_clearance;
  difference() {
    translate([0, 0, base_h + 4])
      rounded_box([tray_w + 8, tray_d + 8, 14], 8);
    translate([0, 0, base_h + 2])
      rounded_box([tray_w, tray_d, 18], 5);
    // Rear cable channel: Type-C power, HDMI, USB.
    translate([-48, tray_d / 2 - 4, base_h + 6])
      cube([96, 24, 18]);
  }
}

module base_body() {
  difference() {
    union() {
      base_outer();
      pi_case_tray();
      // Front core ring, matching the frontend body core.
      translate([0, -body_d / 2 - 1, body_bottom_z + 34])
        y_cylinder(d = 34, h = 9);
      translate([0, -body_d / 2 - 6, body_bottom_z + 34])
        y_cylinder(d = 20, h = 6);
    }

    // Hollow lower body enough for the official Orange Pi case and wires.
    translate([0, 0, base_h + 2])
      rounded_box([pi_case_w + 16, pi_case_d + 16, pi_case_h + 14], 8);

    // Rear slide-in/service opening.
    translate([-62, body_d / 2 - wall - 2, base_h + 6])
      cube([124, wall + 18, 46]);

    // Front lower smile-like cable/vent slot, kept subtle.
    translate([-34, -body_d / 2 - 8, body_bottom_z + 14])
      cube([68, 18, 8]);

    // Bottom ventilation.
    for (x = [-54 : 18 : 54])
      translate([x, -42, -1])
        cube([8, 84, base_h + 3]);

    // Core lens recess.
    translate([0, -body_d / 2 - 10, body_bottom_z + 34])
      y_cylinder(d = 17, h = 20);
  }
}

module assembly_preview() {
  color("#eef2f7") front_head_shell();
  color("#d7e2ea") back_head_shell();
  color("#20242b") base_body();

  // Visual hardware previews only.
  color("#ff4d9d", 0.42)
    translate([-screen_window_w / 2, -head_d / 2 - 1, screen_center_z - screen_window_h / 2])
      cube([screen_window_w, 3, screen_window_h]);
  color("#44d9ff", 0.28)
    translate([-pi_case_w / 2, -pi_case_d / 2, base_h + 8])
      cube([pi_case_w, pi_case_d, pi_case_h]);
  color("#ffd166", 0.35)
    translate([-m260c_board_w / 2, head_d / 2 - wall - m260c_board_t, head_bottom_z + 45])
      cube([m260c_board_w, m260c_board_t, m260c_board_h]);
}

if (PART_ID == 1) {
  front_head_shell();
} else if (PART_ID == 2) {
  back_head_shell();
} else if (PART_ID == 3) {
  base_body();
} else {
  assembly_preview();
}
