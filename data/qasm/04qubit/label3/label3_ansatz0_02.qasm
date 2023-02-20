OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.47932393391271033) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.4245931749102098) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09301608546452754) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.7720020220105653) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.43359617202969414) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.05669787872358585) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3079309538935678) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.31795890954579326) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.41220957955826665) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.6191452747070816) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.4226985143888005) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.4011380731615013) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.13406943284663414) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.33014978079278595) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(-0.31199873544869505) q[3];
cx q[1],q[3];
rx(1.1910192127818033) q[0];
rz(0.5698380559413093) q[0];
rx(-0.5502632267259725) q[1];
rz(-0.00308984462544573) q[1];
rx(-0.962748606793777) q[2];
rz(0.21841999500278558) q[2];
rx(0.37819033799339963) q[3];
rz(0.2372040244158206) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.26761385046759856) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12323498746073587) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1176586537902138) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.4843049097024069) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.07020828300240298) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.019283582882981595) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.555703153767669) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.8557600805467133) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1563828703103053) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.32317976253392655) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(0.8553121163651225) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(-0.07389993450517043) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.6281708045113713) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.7813332832563163) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.3448437855837255) q[3];
cx q[1],q[3];
rx(0.7894991740616831) q[0];
rz(0.32632443996696736) q[0];
rx(-0.5177553791684981) q[1];
rz(0.12442867719013188) q[1];
rx(-0.9751758882616385) q[2];
rz(0.21199607053608735) q[2];
rx(0.34561229044415237) q[3];
rz(0.5913194127171202) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.34257753307755084) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-1.1057612986340994) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.15054039068407143) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.10216844083959196) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.54624948470277) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11836807571515692) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(1.178161205954783) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.4393453698070378) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.4470850667742727) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.782236939904088) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(1.1353892845768294) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.21186351617558347) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-0.9155984744672234) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.7378744027282598) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.27707844804432447) q[3];
cx q[1],q[3];
rx(0.7781424275495344) q[0];
rz(0.16756443473667762) q[0];
rx(-0.5653364603716702) q[1];
rz(-0.29613576001879716) q[1];
rx(-1.162244498161679) q[2];
rz(-0.6940701739255151) q[2];
rx(0.2787845796583964) q[3];
rz(0.01175547745726559) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.8795071366966984) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.9808015134831028) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.1958939019299464) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4688325769499464) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.25868894505179163) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1669758145726802) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(1.674508318446297) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.5894644390303705) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03178862275215078) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-0.950117222757919) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(1.1292210122829662) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.86329549045121) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-1.8257029086692145) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(0.9270790936025403) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.18619213765794868) q[3];
cx q[1],q[3];
rx(0.6636119440619125) q[0];
rz(-0.027621235826411347) q[0];
rx(-0.3088007627898336) q[1];
rz(-0.3223378604569591) q[1];
rx(-0.9824534143281789) q[2];
rz(-0.4720280294738234) q[2];
rx(0.3872798516091226) q[3];
rz(-0.34383803332437535) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.35142839612877425) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.499849174559856) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.3858184543072974) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1769140822554326) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.6841929915320858) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.15200735954629216) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(1.3599291661551345) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.4851084379656296) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.12308535566596789) q[3];
cx q[2],q[3];
h q[0];
h q[2];
cx q[0],q[2];
rz(-1.1493334406356328) q[2];
cx q[0],q[2];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[2];
rz(1.208575543468999) q[2];
cx q[0],q[2];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[2];
rz(0.18255698214842306) q[2];
cx q[0],q[2];
h q[1];
h q[3];
cx q[1],q[3];
rz(-2.1294979553147657) q[3];
cx q[1],q[3];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[3];
rz(1.3082873779979314) q[3];
cx q[1],q[3];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[1],q[3];
rz(0.18467702094561142) q[3];
cx q[1],q[3];
rx(0.3667576520890337) q[0];
rz(-0.04004817149609198) q[0];
rx(-0.3490129762801367) q[1];
rz(-0.41784323465397505) q[1];
rx(-0.3886941431790828) q[2];
rz(-0.11036649583686553) q[2];
rx(-0.3246872941292972) q[3];
rz(0.327784729381483) q[3];