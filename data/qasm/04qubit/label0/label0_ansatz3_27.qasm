OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.5866100337901785) q[0];
rz(-2.1498608135268285) q[0];
ry(0.7273327750804021) q[1];
rz(-3.1179881999386874) q[1];
ry(0.5968552616934284) q[2];
rz(-0.4566161708900586) q[2];
ry(-2.818540146341136) q[3];
rz(2.237446559872448) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8798966572198168) q[0];
rz(-1.1915228583710622) q[0];
ry(-2.284249188987692) q[1];
rz(-2.5792352919237533) q[1];
ry(3.1099652341768063) q[2];
rz(-1.5911642561218038) q[2];
ry(-0.2087864448854764) q[3];
rz(-1.3369276062893394) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.7245696581146638) q[0];
rz(-2.5962494400255323) q[0];
ry(-1.215595171725731) q[1];
rz(-1.9403299581709161) q[1];
ry(-0.3207611117573583) q[2];
rz(-1.5003478612110959) q[2];
ry(-0.2770249738870267) q[3];
rz(1.6673377199054422) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.09960871477256) q[0];
rz(-2.4585451475258178) q[0];
ry(-1.909835522575503) q[1];
rz(-0.8813068552178869) q[1];
ry(0.5083990391822333) q[2];
rz(1.9786409016523985) q[2];
ry(-1.2874095942323498) q[3];
rz(-2.2729683923511166) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.105873984150016) q[0];
rz(2.566045555118333) q[0];
ry(-1.1199461769706593) q[1];
rz(1.1574966536391755) q[1];
ry(0.32364065808177067) q[2];
rz(0.09591160265247836) q[2];
ry(-1.6608517399926532) q[3];
rz(3.0293760105168976) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9631539811457879) q[0];
rz(-2.3850713413856104) q[0];
ry(2.0704659749698084) q[1];
rz(-0.5594899788601618) q[1];
ry(-2.2235515304955547) q[2];
rz(0.07795875621621386) q[2];
ry(2.495634303504171) q[3];
rz(-0.7039179212815982) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.031945264851655) q[0];
rz(2.3174621255497123) q[0];
ry(-2.4133345056152296) q[1];
rz(1.3640637354475107) q[1];
ry(-1.0934057829971078) q[2];
rz(0.05758564474821348) q[2];
ry(-1.5626723076085796) q[3];
rz(1.9448874797973659) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.3220635435274009) q[0];
rz(-1.0539603677054323) q[0];
ry(0.639154342964412) q[1];
rz(-2.7086268658814046) q[1];
ry(-1.273819166138851) q[2];
rz(-2.7686453552588444) q[2];
ry(1.1379940791997467) q[3];
rz(0.5380240508427727) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.0632658840295033) q[0];
rz(-2.0760565797148374) q[0];
ry(-2.483073019716863) q[1];
rz(0.47193751148103225) q[1];
ry(2.9034713367701124) q[2];
rz(-2.101109620809871) q[2];
ry(-2.0208671181631632) q[3];
rz(-0.2501589681951817) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.0344112349876236) q[0];
rz(-2.588683542935214) q[0];
ry(-0.71703188478328) q[1];
rz(-0.7026301648084496) q[1];
ry(-2.5680133964349654) q[2];
rz(0.1731163856235609) q[2];
ry(-2.5441975262322147) q[3];
rz(-0.13540831227423103) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.127977483988602) q[0];
rz(-1.3367300341210582) q[0];
ry(-1.9084344609110917) q[1];
rz(-2.4916868747726784) q[1];
ry(2.2966375963678045) q[2];
rz(1.9800094817506384) q[2];
ry(2.055214254471678) q[3];
rz(-1.8556176572621927) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.958979997113354) q[0];
rz(1.8864451401993145) q[0];
ry(2.2624358272212053) q[1];
rz(-0.9174299075019173) q[1];
ry(1.8224324270603243) q[2];
rz(-0.6035423884784069) q[2];
ry(2.105778569293836) q[3];
rz(-1.8750793417215301) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0418387663393043) q[0];
rz(1.4997136259466435) q[0];
ry(1.0012945241734563) q[1];
rz(2.8978216428817536) q[1];
ry(-1.3663880694647477) q[2];
rz(0.1641208899389483) q[2];
ry(-2.903450863205582) q[3];
rz(0.7589040261181622) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8303425861422582) q[0];
rz(-2.839576289745057) q[0];
ry(1.990860669227338) q[1];
rz(1.1981199148072967) q[1];
ry(-1.4258838470938278) q[2];
rz(-2.444674040032746) q[2];
ry(1.1753468878514894) q[3];
rz(-3.110920648564341) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0003702136869106) q[0];
rz(-0.5946588090177698) q[0];
ry(-2.7276674929022673) q[1];
rz(-1.3493728202348363) q[1];
ry(3.0269447052293614) q[2];
rz(-2.309654107397747) q[2];
ry(-2.828349265602407) q[3];
rz(-0.45729703339021643) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.728461617181547) q[0];
rz(-2.989898409455044) q[0];
ry(-1.7774727343578007) q[1];
rz(0.6049397930098865) q[1];
ry(1.8014672560232325) q[2];
rz(-2.767471882795783) q[2];
ry(-1.5573077275018574) q[3];
rz(2.476483490520516) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.520040301859463) q[0];
rz(2.172410897312021) q[0];
ry(-2.6131305281255304) q[1];
rz(1.9871647927532223) q[1];
ry(-1.2299637042819977) q[2];
rz(-1.090197214802778) q[2];
ry(-0.9558001609359917) q[3];
rz(-2.0044184307185047) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.029524390087736715) q[0];
rz(0.4924185226444573) q[0];
ry(-0.06538832834553691) q[1];
rz(0.5085952945051893) q[1];
ry(-1.7597080739744255) q[2];
rz(-2.5818601415966267) q[2];
ry(2.490487967348612) q[3];
rz(-2.902041800289099) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.09586094105076448) q[0];
rz(0.9403831089971236) q[0];
ry(-2.0759460892161687) q[1];
rz(-2.0858401521966696) q[1];
ry(-2.6279352722106286) q[2];
rz(-0.28880923849684764) q[2];
ry(0.5217745385182881) q[3];
rz(3.022058233468467) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.1327265791200576) q[0];
rz(-1.33283887629443) q[0];
ry(-0.23674522769951434) q[1];
rz(-0.6902850664519279) q[1];
ry(1.6131510657849322) q[2];
rz(-1.9230655807330177) q[2];
ry(-0.7623493930123582) q[3];
rz(0.5098888688546754) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.884800930801264) q[0];
rz(-2.1107824789802003) q[0];
ry(-3.085987869332587) q[1];
rz(-0.9612408931538399) q[1];
ry(-0.01836136456378412) q[2];
rz(1.5672532786423579) q[2];
ry(0.34814411389019456) q[3];
rz(0.6997120358845181) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.02159891417879667) q[0];
rz(-3.080641387507373) q[0];
ry(2.941757764122706) q[1];
rz(2.948964377858907) q[1];
ry(2.3000346979281217) q[2];
rz(0.30472453860669524) q[2];
ry(-1.1738029761776057) q[3];
rz(0.9290280653749274) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.3244958899253327) q[0];
rz(-0.4345362372801462) q[0];
ry(-1.6216096451171254) q[1];
rz(0.1606818444532541) q[1];
ry(1.3155496113061869) q[2];
rz(2.666824352241521) q[2];
ry(2.7328953241039304) q[3];
rz(-2.152704680564396) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.39882335062423535) q[0];
rz(-1.0385671974961879) q[0];
ry(-0.118028668918866) q[1];
rz(-0.8343913824698052) q[1];
ry(2.176401408105151) q[2];
rz(-1.6535311778232917) q[2];
ry(3.076796001231259) q[3];
rz(2.746844594463976) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2349924063688176) q[0];
rz(-1.0368276200409254) q[0];
ry(-2.6591616800765556) q[1];
rz(-0.7331162800102723) q[1];
ry(2.330947323468471) q[2];
rz(-0.9021330118593639) q[2];
ry(-0.7895652668354842) q[3];
rz(2.0636905774077237) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.2333830681643154) q[0];
rz(2.9142689765294265) q[0];
ry(0.49957520183216797) q[1];
rz(-0.08636859951840892) q[1];
ry(-3.036344049537391) q[2];
rz(0.24708172599843617) q[2];
ry(0.16503781088711286) q[3];
rz(2.404200090134393) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.757310687146143) q[0];
rz(-1.4692509221771408) q[0];
ry(-2.1691535939927657) q[1];
rz(-2.638734673276175) q[1];
ry(-0.7208174450960563) q[2];
rz(-2.329029916117571) q[2];
ry(0.24492030750901117) q[3];
rz(-0.8814130221664325) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.9408254562413677) q[0];
rz(1.3570515703790573) q[0];
ry(2.9622987942784684) q[1];
rz(2.1473221736180235) q[1];
ry(0.6733700101057449) q[2];
rz(2.5031141537965285) q[2];
ry(-0.2881860124568394) q[3];
rz(-1.6357387807526989) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.9271627310489012) q[0];
rz(-0.41338269490373314) q[0];
ry(1.1498312177417596) q[1];
rz(-0.7850136286143294) q[1];
ry(1.8146540064179726) q[2];
rz(3.0629336899041326) q[2];
ry(-0.9953686354440164) q[3];
rz(2.579569216727457) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.4433790710545829) q[0];
rz(3.0607170398805783) q[0];
ry(1.1419156265109314) q[1];
rz(-1.5164789402249461) q[1];
ry(-0.9397365539861366) q[2];
rz(1.8488935335335412) q[2];
ry(2.9972797729076235) q[3];
rz(-2.3157877118980568) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.0051259250811042) q[0];
rz(-2.1662289531161276) q[0];
ry(-0.2838808153719379) q[1];
rz(2.755908622465884) q[1];
ry(1.7396699465427226) q[2];
rz(-0.4818672528476587) q[2];
ry(-2.1979748592683794) q[3];
rz(1.2481329580346658) q[3];