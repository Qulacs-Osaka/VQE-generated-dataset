OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.1780018883387688) q[0];
ry(0.6390405147578093) q[1];
cx q[0],q[1];
ry(1.0687906924584532) q[0];
ry(0.41558781091856156) q[1];
cx q[0],q[1];
ry(-1.513070801850888) q[2];
ry(0.8260376611042073) q[3];
cx q[2],q[3];
ry(-1.3655193618919998) q[2];
ry(1.512826920767334) q[3];
cx q[2],q[3];
ry(-0.8902200610621209) q[0];
ry(2.542881696470153) q[2];
cx q[0],q[2];
ry(2.103975021480984) q[0];
ry(0.6142670924497251) q[2];
cx q[0],q[2];
ry(-2.883640486263304) q[1];
ry(1.5609162560706666) q[3];
cx q[1],q[3];
ry(-2.12236051305467) q[1];
ry(1.8140882410215502) q[3];
cx q[1],q[3];
ry(-1.798547087317039) q[0];
ry(-2.1044472357486885) q[1];
cx q[0],q[1];
ry(-2.9151933344168564) q[0];
ry(-2.1718638697393926) q[1];
cx q[0],q[1];
ry(1.824265366248876) q[2];
ry(0.002365129275814226) q[3];
cx q[2],q[3];
ry(1.2648329137017025) q[2];
ry(0.6839109072651988) q[3];
cx q[2],q[3];
ry(0.053654755261620106) q[0];
ry(-1.4293083743065704) q[2];
cx q[0],q[2];
ry(-0.05694336724767215) q[0];
ry(-2.3072374637663526) q[2];
cx q[0],q[2];
ry(-2.168054100257846) q[1];
ry(-1.316254038996646) q[3];
cx q[1],q[3];
ry(0.06943688307267597) q[1];
ry(1.0869816444145366) q[3];
cx q[1],q[3];
ry(-0.5367361405122315) q[0];
ry(2.797043528931597) q[1];
cx q[0],q[1];
ry(1.0277330798488125) q[0];
ry(3.133423185304981) q[1];
cx q[0],q[1];
ry(-0.25330916238412815) q[2];
ry(-1.1994662366227318) q[3];
cx q[2],q[3];
ry(3.030365904404914) q[2];
ry(-2.3624754017807463) q[3];
cx q[2],q[3];
ry(-2.687122760246642) q[0];
ry(-2.558388221123413) q[2];
cx q[0],q[2];
ry(-0.11325526602528946) q[0];
ry(-0.2028187825975688) q[2];
cx q[0],q[2];
ry(-0.9065283736615416) q[1];
ry(-0.4481070893544032) q[3];
cx q[1],q[3];
ry(-3.1048506894950787) q[1];
ry(-2.1600151936877063) q[3];
cx q[1],q[3];
ry(-3.123501622323846) q[0];
ry(-1.7752014846277904) q[1];
cx q[0],q[1];
ry(2.209893224344547) q[0];
ry(0.5650242400084561) q[1];
cx q[0],q[1];
ry(-2.3192323062906235) q[2];
ry(-2.877763937853357) q[3];
cx q[2],q[3];
ry(-2.978767185231545) q[2];
ry(-3.0276579637488887) q[3];
cx q[2],q[3];
ry(-2.0256870575013695) q[0];
ry(2.3930392879385143) q[2];
cx q[0],q[2];
ry(-0.575894169548177) q[0];
ry(0.5111607866314865) q[2];
cx q[0],q[2];
ry(-1.4503081868989736) q[1];
ry(2.186179074398079) q[3];
cx q[1],q[3];
ry(1.1206222164858173) q[1];
ry(0.9234411910964635) q[3];
cx q[1],q[3];
ry(2.385661054722988) q[0];
ry(-1.2232896306458096) q[1];
cx q[0],q[1];
ry(-2.084043581280907) q[0];
ry(-2.619257699360632) q[1];
cx q[0],q[1];
ry(-2.578985769173159) q[2];
ry(-2.314906927365464) q[3];
cx q[2],q[3];
ry(2.6708518374784402) q[2];
ry(2.088610421578406) q[3];
cx q[2],q[3];
ry(-1.717103404626294) q[0];
ry(0.7596960726118125) q[2];
cx q[0],q[2];
ry(2.835501500103171) q[0];
ry(-1.0110903538797071) q[2];
cx q[0],q[2];
ry(2.3319725877900765) q[1];
ry(-0.6616291565105739) q[3];
cx q[1],q[3];
ry(0.5173695035531658) q[1];
ry(2.811934260753829) q[3];
cx q[1],q[3];
ry(-0.4201707143626914) q[0];
ry(-2.857664520858831) q[1];
cx q[0],q[1];
ry(1.1958273863155273) q[0];
ry(0.9056649595507563) q[1];
cx q[0],q[1];
ry(0.4248895454757502) q[2];
ry(0.9065680227759048) q[3];
cx q[2],q[3];
ry(-0.6236975278736159) q[2];
ry(1.0352983730496765) q[3];
cx q[2],q[3];
ry(2.7021329316371063) q[0];
ry(0.8935265684840124) q[2];
cx q[0],q[2];
ry(-0.1760805003674788) q[0];
ry(-1.404099134740137) q[2];
cx q[0],q[2];
ry(-2.0533288687702296) q[1];
ry(0.9906217488200612) q[3];
cx q[1],q[3];
ry(1.2059716970949) q[1];
ry(-1.722772017436276) q[3];
cx q[1],q[3];
ry(-2.419543001174957) q[0];
ry(2.956019089486947) q[1];
cx q[0],q[1];
ry(0.5320829807844819) q[0];
ry(1.8662078729815734) q[1];
cx q[0],q[1];
ry(-2.061832980251197) q[2];
ry(-1.6740044147453927) q[3];
cx q[2],q[3];
ry(-2.605658270413211) q[2];
ry(-1.8092663842559302) q[3];
cx q[2],q[3];
ry(0.2714893892751382) q[0];
ry(1.287387260011533) q[2];
cx q[0],q[2];
ry(0.11889731792494543) q[0];
ry(-1.9523847256690514) q[2];
cx q[0],q[2];
ry(-2.3580165405084905) q[1];
ry(-0.3781716686411958) q[3];
cx q[1],q[3];
ry(-2.6982280534051015) q[1];
ry(-1.414734640463575) q[3];
cx q[1],q[3];
ry(-0.1168321768081897) q[0];
ry(1.0152829106829149) q[1];
cx q[0],q[1];
ry(-1.7037250445022292) q[0];
ry(0.16556388893679155) q[1];
cx q[0],q[1];
ry(-0.9428216874057069) q[2];
ry(-0.19522679952034494) q[3];
cx q[2],q[3];
ry(-2.431766107587008) q[2];
ry(2.214800542365502) q[3];
cx q[2],q[3];
ry(-0.54521992677064) q[0];
ry(0.3259443554875884) q[2];
cx q[0],q[2];
ry(-0.07563663883593376) q[0];
ry(-0.5322172652126356) q[2];
cx q[0],q[2];
ry(0.28567311558297076) q[1];
ry(2.0447891870667867) q[3];
cx q[1],q[3];
ry(1.9373946365375025) q[1];
ry(-0.5791977346586927) q[3];
cx q[1],q[3];
ry(-2.8507448275481546) q[0];
ry(-2.929808629964707) q[1];
ry(1.1381822931541885) q[2];
ry(0.23689222558648132) q[3];