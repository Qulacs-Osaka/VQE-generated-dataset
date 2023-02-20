OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.4269388447369504) q[0];
rz(1.6730422053649114) q[0];
ry(-1.4900457939915823) q[1];
rz(-2.219738327183644) q[1];
ry(1.9497881734559839) q[2];
rz(-1.6162526730589948) q[2];
ry(0.28050217405842315) q[3];
rz(0.974280796772114) q[3];
ry(2.23297110146855) q[4];
rz(-2.2666107547789607) q[4];
ry(1.0486356752552695) q[5];
rz(-2.9579320193529726) q[5];
ry(2.24669409088939) q[6];
rz(2.1718740590532684) q[6];
ry(2.447086503693656) q[7];
rz(-0.8443121955869826) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.7191844145263206) q[0];
rz(0.6203125278599956) q[0];
ry(2.7362515114737707) q[1];
rz(0.21574755344418062) q[1];
ry(-1.7260431085610177) q[2];
rz(2.6460510638528496) q[2];
ry(-0.41539803445916235) q[3];
rz(-1.3384975477995518) q[3];
ry(-0.3973607441688553) q[4];
rz(2.4601329791617546) q[4];
ry(0.6173895370572735) q[5];
rz(-1.3121882956836401) q[5];
ry(-0.6688870854874633) q[6];
rz(1.7366711698530606) q[6];
ry(-0.5362576803177557) q[7];
rz(-2.880180422988956) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.174378228071488) q[0];
rz(-2.0605559584075195) q[0];
ry(-1.6362949661536916) q[1];
rz(-2.184021247739283) q[1];
ry(-1.5365804583844147) q[2];
rz(-0.9461154562782019) q[2];
ry(-2.636705127215702) q[3];
rz(-3.0495415954716467) q[3];
ry(2.6348532961103945) q[4];
rz(1.5123574936545987) q[4];
ry(-3.007815085340729) q[5];
rz(-2.7328927393466853) q[5];
ry(1.7167556228186625) q[6];
rz(-1.8124181391133076) q[6];
ry(1.505259663063301) q[7];
rz(-1.8345995402743807) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.514920464643828) q[0];
rz(-1.80124661991712) q[0];
ry(-0.9854768124953299) q[1];
rz(-0.4818870448628055) q[1];
ry(1.1669372900472021) q[2];
rz(1.29446027909556) q[2];
ry(-1.0012949083635398) q[3];
rz(-0.24709092452668996) q[3];
ry(-0.27764302701936483) q[4];
rz(1.9673137692999116) q[4];
ry(-1.4451163220846566) q[5];
rz(-0.143517762946704) q[5];
ry(-2.8175674179439496) q[6];
rz(-1.4947043929174746) q[6];
ry(1.080387371051675) q[7];
rz(1.0477557626229175) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.7539145488443832) q[0];
rz(2.286884284987292) q[0];
ry(-2.1744598354632405) q[1];
rz(2.3609523947779745) q[1];
ry(2.0885400302835126) q[2];
rz(-0.27962892408895534) q[2];
ry(0.24104911446914026) q[3];
rz(1.7274262187219156) q[3];
ry(-1.2300582779544982) q[4];
rz(-0.1104170639497179) q[4];
ry(2.2031669962107676) q[5];
rz(-2.50343608758967) q[5];
ry(-0.4173322855227997) q[6];
rz(2.554455987639896) q[6];
ry(-0.32325223499578914) q[7];
rz(1.1712527111044557) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.0352131078875644) q[0];
rz(-0.17978460448101075) q[0];
ry(-2.4989776601270144) q[1];
rz(-1.681230288045568) q[1];
ry(0.3306365659380035) q[2];
rz(0.14498106772118827) q[2];
ry(2.8206307346069495) q[3];
rz(2.8093574302188604) q[3];
ry(-1.851224947045296) q[4];
rz(1.0447368412689944) q[4];
ry(-1.9547811823525443) q[5];
rz(0.8458002352083183) q[5];
ry(2.6119313669560107) q[6];
rz(-2.0627468028382405) q[6];
ry(-1.548503689966535) q[7];
rz(0.016364924464425976) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.38019697156218424) q[0];
rz(3.056694764320806) q[0];
ry(-2.846283309835462) q[1];
rz(2.043126479757569) q[1];
ry(2.702025713787972) q[2];
rz(-0.634947138700941) q[2];
ry(0.7590621571195911) q[3];
rz(3.0548997250782888) q[3];
ry(-2.888270266916118) q[4];
rz(0.4624559452699444) q[4];
ry(-2.2256546949543425) q[5];
rz(2.4709626861513967) q[5];
ry(0.7393898063470239) q[6];
rz(-1.0120727299574988) q[6];
ry(1.7271821903092575) q[7];
rz(-1.5691708639850763) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.7998613977256) q[0];
rz(-3.0113861019406616) q[0];
ry(-1.0837911455737026) q[1];
rz(-2.3116474834045757) q[1];
ry(0.387983495292393) q[2];
rz(0.4687083969794896) q[2];
ry(-2.3441222935407247) q[3];
rz(1.3253116171478894) q[3];
ry(-2.7737567785208115) q[4];
rz(0.19318773599615793) q[4];
ry(0.2923686514358989) q[5];
rz(2.32853529727957) q[5];
ry(0.31427346632739184) q[6];
rz(-0.92250267185588) q[6];
ry(-1.4845307618828385) q[7];
rz(-0.13024852325468483) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.5471080174826817) q[0];
rz(2.020251796522735) q[0];
ry(0.15410771762834205) q[1];
rz(0.687590064111392) q[1];
ry(-2.5521838718804677) q[2];
rz(2.4244663378305007) q[2];
ry(-1.9924982993241214) q[3];
rz(1.22730687156659) q[3];
ry(3.110515789734488) q[4];
rz(-2.4520924046929666) q[4];
ry(-2.203744252711676) q[5];
rz(1.5596596292628442) q[5];
ry(-1.5712873280418709) q[6];
rz(2.445316972758031) q[6];
ry(-1.7220312112989813) q[7];
rz(-2.9297431480151297) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.572220381855535) q[0];
rz(0.7663465247075241) q[0];
ry(-1.552579130713295) q[1];
rz(1.83151543966524) q[1];
ry(0.5934530802880786) q[2];
rz(-2.017583628249378) q[2];
ry(-2.673322743087324) q[3];
rz(-1.051173210636412) q[3];
ry(2.9873237705867726) q[4];
rz(-2.1165847686630936) q[4];
ry(0.05805369766074407) q[5];
rz(-2.7255149992498118) q[5];
ry(1.3548947844655772) q[6];
rz(0.8233231585039249) q[6];
ry(-2.43822059144697) q[7];
rz(-1.2638398518509666) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.4170069929017997) q[0];
rz(2.4728114931203646) q[0];
ry(0.5859386172335919) q[1];
rz(2.1675346601568455) q[1];
ry(-2.100754735357671) q[2];
rz(-2.857009476676857) q[2];
ry(-0.7829822854379497) q[3];
rz(-2.86664250988092) q[3];
ry(-1.7231914740069243) q[4];
rz(2.9115113463234628) q[4];
ry(-2.909054339677628) q[5];
rz(-1.3730525937680564) q[5];
ry(-0.9220845381453885) q[6];
rz(-2.920829364475229) q[6];
ry(1.9519583037743473) q[7];
rz(-0.8185599999252958) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.023915365857803) q[0];
rz(-0.49994620053661704) q[0];
ry(-1.233220000415064) q[1];
rz(-1.9246237057676048) q[1];
ry(-2.422969673136759) q[2];
rz(0.9993667296155252) q[2];
ry(-2.8553402990978904) q[3];
rz(-2.2526876282967283) q[3];
ry(-1.57170823683032) q[4];
rz(-1.0471027220200713) q[4];
ry(-1.3614799967734825) q[5];
rz(-1.2398984247864517) q[5];
ry(-0.4882493241662784) q[6];
rz(0.3451362656557473) q[6];
ry(-0.14057353090421998) q[7];
rz(-0.819526798367742) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.41182519070581636) q[0];
rz(0.20311888596361438) q[0];
ry(-0.4330222828295457) q[1];
rz(1.0880732450762665) q[1];
ry(2.3237259345618213) q[2];
rz(1.8206629150581828) q[2];
ry(-0.3942601536918664) q[3];
rz(-2.806568264786039) q[3];
ry(1.1110789391849922) q[4];
rz(2.2934228051512173) q[4];
ry(-1.0159023441237247) q[5];
rz(1.480136032278707) q[5];
ry(-2.571553115062408) q[6];
rz(2.7969680265672934) q[6];
ry(-0.8957029516976855) q[7];
rz(1.4444390262738114) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.1601264225769992) q[0];
rz(-1.754926374074885) q[0];
ry(1.9398242397932508) q[1];
rz(2.4683259558027397) q[1];
ry(0.8094662330079014) q[2];
rz(0.24341445047090038) q[2];
ry(2.1628472351871064) q[3];
rz(1.490910988290075) q[3];
ry(0.5027037868773448) q[4];
rz(-0.3708421761866588) q[4];
ry(-2.6581642742180804) q[5];
rz(-1.641702153484018) q[5];
ry(-1.6882318385519657) q[6];
rz(-3.0137234995590307) q[6];
ry(1.8505857692117482) q[7];
rz(-1.1130249530252274) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.572849849434866) q[0];
rz(-0.13798726786117685) q[0];
ry(0.5933793121854923) q[1];
rz(0.7417731967246022) q[1];
ry(-1.1335156893102585) q[2];
rz(2.0858261320693066) q[2];
ry(1.990295434892289) q[3];
rz(-1.8556903810561645) q[3];
ry(1.5095827554291539) q[4];
rz(-3.008264704887679) q[4];
ry(0.9309168367029574) q[5];
rz(2.2727340871511457) q[5];
ry(1.0791454889142145) q[6];
rz(-3.1355253361081115) q[6];
ry(-0.5996021399518883) q[7];
rz(2.274484634795777) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.1164193549739565) q[0];
rz(-3.1158692329007875) q[0];
ry(0.15023959813060728) q[1];
rz(-1.590552869455728) q[1];
ry(-1.4544661835809825) q[2];
rz(1.627012284790377) q[2];
ry(-1.4553526537928048) q[3];
rz(2.61867645049249) q[3];
ry(1.6098621019904311) q[4];
rz(0.9737455667783403) q[4];
ry(1.2814427018757426) q[5];
rz(0.7339705319513208) q[5];
ry(2.4404620789860365) q[6];
rz(-0.15676387537036438) q[6];
ry(-2.1081911514608) q[7];
rz(-0.06068285664380091) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.6446314183954156) q[0];
rz(-0.20865539932919835) q[0];
ry(-2.137401744881552) q[1];
rz(2.044354127964251) q[1];
ry(-0.5576016949511774) q[2];
rz(-1.1934701168476591) q[2];
ry(-2.227068536717761) q[3];
rz(2.1652753827479) q[3];
ry(0.6266776318910001) q[4];
rz(2.0056807213651684) q[4];
ry(-1.0101229304086035) q[5];
rz(1.058552885784222) q[5];
ry(2.316087387856531) q[6];
rz(-2.436908049904627) q[6];
ry(-0.2804134644780658) q[7];
rz(1.9738382387716815) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.7193685418021103) q[0];
rz(1.6031778959572331) q[0];
ry(2.9841716480779565) q[1];
rz(2.102905244039179) q[1];
ry(0.8299424039277873) q[2];
rz(-0.9477235586606163) q[2];
ry(-2.932853919548808) q[3];
rz(-0.07929473307451913) q[3];
ry(-0.292773674800511) q[4];
rz(1.505271314258585) q[4];
ry(-0.5861059654236698) q[5];
rz(0.42104424954930136) q[5];
ry(2.396031178403752) q[6];
rz(-2.5421446423721106) q[6];
ry(1.8518635549420415) q[7];
rz(-1.6835010324610764) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.5635230899279489) q[0];
rz(2.528054787059722) q[0];
ry(2.1834205528753654) q[1];
rz(0.9295197517295) q[1];
ry(-0.5185980040106202) q[2];
rz(-0.9889494354001985) q[2];
ry(0.6205618412697209) q[3];
rz(1.3382795260650937) q[3];
ry(-2.5029307986975597) q[4];
rz(-1.4427011350278995) q[4];
ry(0.2976642136423484) q[5];
rz(0.7169694491357531) q[5];
ry(-1.1583170128689249) q[6];
rz(-2.386173744716591) q[6];
ry(-1.7384623636656) q[7];
rz(1.5374925579159342) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.48629070277877334) q[0];
rz(2.144190798853637) q[0];
ry(2.613009999392854) q[1];
rz(-0.2320116212727867) q[1];
ry(1.223340203905011) q[2];
rz(-2.434105695288075) q[2];
ry(-1.7191589227676864) q[3];
rz(-2.740141467402137) q[3];
ry(-0.62143201598186) q[4];
rz(1.5070908184378702) q[4];
ry(-2.3402324608117278) q[5];
rz(1.603283616732984) q[5];
ry(-1.116647715049944) q[6];
rz(-1.0164882843014622) q[6];
ry(2.4504077955773473) q[7];
rz(1.0362824509926807) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.968501505276697) q[0];
rz(2.7818065454908267) q[0];
ry(2.7437302478358214) q[1];
rz(1.075980500954898) q[1];
ry(1.8032273040971685) q[2];
rz(-1.917647902651963) q[2];
ry(2.251542831155391) q[3];
rz(-2.8587142958593894) q[3];
ry(-1.4252600220912077) q[4];
rz(-2.4524217462354025) q[4];
ry(-2.062831566243505) q[5];
rz(-2.6621587939224023) q[5];
ry(1.5663459122210661) q[6];
rz(-2.614638225389091) q[6];
ry(-1.5554789861432996) q[7];
rz(-0.9275275661376475) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.415265292261546) q[0];
rz(2.7549551418233937) q[0];
ry(-1.8581562992124492) q[1];
rz(-1.262406966770249) q[1];
ry(-2.8439089114190295) q[2];
rz(-0.056506394385029246) q[2];
ry(-2.425934137535728) q[3];
rz(2.383368781813964) q[3];
ry(2.3621596425905644) q[4];
rz(3.0414023042699756) q[4];
ry(2.3702925247568603) q[5];
rz(0.3451873050454959) q[5];
ry(-1.1875628833964629) q[6];
rz(-3.060226453610941) q[6];
ry(-0.43194455658089126) q[7];
rz(0.8600445485396402) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.9458628779963414) q[0];
rz(-2.608409240226075) q[0];
ry(-1.4308268019476698) q[1];
rz(-0.30228561647120866) q[1];
ry(1.4231416399061645) q[2];
rz(-2.3007233779738745) q[2];
ry(-0.7460650965804201) q[3];
rz(2.8011173980765602) q[3];
ry(-0.7887025907877827) q[4];
rz(-0.5573839915322363) q[4];
ry(-0.7399860014945842) q[5];
rz(-1.5879299114789303) q[5];
ry(1.571513678510256) q[6];
rz(2.1407112972105953) q[6];
ry(1.6624898336884044) q[7];
rz(-2.1870315241197895) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.8234190400845343) q[0];
rz(1.0971218142560764) q[0];
ry(1.7861451566745543) q[1];
rz(-0.6593785678216467) q[1];
ry(-1.3490522802265055) q[2];
rz(1.8861295695101568) q[2];
ry(-0.5776650960642042) q[3];
rz(-2.7441777064427706) q[3];
ry(0.35528175229199055) q[4];
rz(-0.5880814591351556) q[4];
ry(2.3303642085903933) q[5];
rz(3.1067442368124376) q[5];
ry(0.3099516865125853) q[6];
rz(2.636419553026912) q[6];
ry(0.5177885323393436) q[7];
rz(0.9559666560504096) q[7];