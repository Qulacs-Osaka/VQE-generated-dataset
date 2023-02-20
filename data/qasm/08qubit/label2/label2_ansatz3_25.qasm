OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.8949194575287209) q[0];
rz(-3.0865938503882746) q[0];
ry(2.5298036806668778) q[1];
rz(1.1956484350005379) q[1];
ry(-1.1320979342484936) q[2];
rz(2.65030866951982) q[2];
ry(-1.8162722825364375) q[3];
rz(1.94955920760375) q[3];
ry(-2.788569400752494) q[4];
rz(-1.203126831759624) q[4];
ry(-1.2696947916876862) q[5];
rz(-2.932852140310429) q[5];
ry(-2.8938000319184054) q[6];
rz(-2.0292479076446526) q[6];
ry(0.7859473164455708) q[7];
rz(-2.1846145779819004) q[7];
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
ry(-2.1383513360423034) q[0];
rz(-0.34357600918855097) q[0];
ry(2.672488760753908) q[1];
rz(2.7989977116693687) q[1];
ry(-2.9197783449024843) q[2];
rz(-0.5944959909050205) q[2];
ry(1.0405250471600125) q[3];
rz(-0.19211711863143766) q[3];
ry(-2.213012931499968) q[4];
rz(1.609981706781081) q[4];
ry(-1.8376649488462353) q[5];
rz(-0.21731661089144175) q[5];
ry(-2.815528337258639) q[6];
rz(1.5967735590448187) q[6];
ry(1.1634775412123801) q[7];
rz(1.612884109037427) q[7];
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
ry(0.13837297210017055) q[0];
rz(2.715582048664295) q[0];
ry(-2.3222739488044497) q[1];
rz(2.275980767420248) q[1];
ry(-1.048968676814395) q[2];
rz(2.8325956808621737) q[2];
ry(0.5930232460523781) q[3];
rz(-0.8036017369915418) q[3];
ry(-1.0596119320080342) q[4];
rz(0.1261531472756472) q[4];
ry(-2.663161460942699) q[5];
rz(0.6753682741016215) q[5];
ry(-0.6848540880516643) q[6];
rz(-1.0659440222014516) q[6];
ry(-0.2890893177327323) q[7];
rz(-1.6425059865580875) q[7];
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
ry(-2.0651810775560566) q[0];
rz(-2.123313585072407) q[0];
ry(1.1242085985139623) q[1];
rz(-0.913720124419059) q[1];
ry(0.9877414128791361) q[2];
rz(-0.5357471239473472) q[2];
ry(-2.8293354165794806) q[3];
rz(2.1803147585134672) q[3];
ry(0.598136674477545) q[4];
rz(-2.3679068702418022) q[4];
ry(1.3484080068314497) q[5];
rz(-2.9663275247325975) q[5];
ry(-3.116862388313478) q[6];
rz(0.5584969722681618) q[6];
ry(2.514677713690348) q[7];
rz(-1.4081938386061785) q[7];
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
ry(0.8620182993470354) q[0];
rz(-2.1582264057469236) q[0];
ry(1.86555931668252) q[1];
rz(-0.17672816280080728) q[1];
ry(3.020543535713468) q[2];
rz(2.3354372432977577) q[2];
ry(-1.3618108207329185) q[3];
rz(2.1502192513472265) q[3];
ry(1.6693087508205666) q[4];
rz(3.043135251803635) q[4];
ry(2.65536810254037) q[5];
rz(-0.3893408022773356) q[5];
ry(2.5136498102017413) q[6];
rz(-1.13495519620794) q[6];
ry(2.300145985505717) q[7];
rz(2.504488313249146) q[7];
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
ry(-1.5303833416520518) q[0];
rz(-2.2524377088488023) q[0];
ry(-1.1092904083146076) q[1];
rz(3.113013265184188) q[1];
ry(-0.6881061718017895) q[2];
rz(0.24520469442491957) q[2];
ry(2.119539417355449) q[3];
rz(-2.823221396725661) q[3];
ry(-1.1980875263166588) q[4];
rz(-0.13578733355656566) q[4];
ry(-2.0719285038091044) q[5];
rz(2.8355705945714296) q[5];
ry(-1.5356503221312687) q[6];
rz(-1.7290968226472705) q[6];
ry(-0.42932966173960385) q[7];
rz(-2.601937885443143) q[7];
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
ry(-2.1839018414548477) q[0];
rz(0.7871161774184083) q[0];
ry(-2.429442757058172) q[1];
rz(-3.0909299507474652) q[1];
ry(-0.5717780035665978) q[2];
rz(-1.7546925199515164) q[2];
ry(-2.8427209871545975) q[3];
rz(2.15760378154575) q[3];
ry(-0.9012293389136544) q[4];
rz(0.16201577323540017) q[4];
ry(1.791208274292136) q[5];
rz(-1.3768864706738295) q[5];
ry(-3.002520834623873) q[6];
rz(2.9984210981322796) q[6];
ry(0.36655889666795644) q[7];
rz(1.404201273222012) q[7];
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
ry(0.33807565464411216) q[0];
rz(1.2819449146852433) q[0];
ry(0.7460214847649421) q[1];
rz(3.072217646516795) q[1];
ry(-1.903390153283091) q[2];
rz(1.3862533252061884) q[2];
ry(-1.5104033604349065) q[3];
rz(2.9056918738835344) q[3];
ry(1.8043296940036457) q[4];
rz(-1.2925662574081658) q[4];
ry(-1.2934044283279151) q[5];
rz(-0.2521619676873719) q[5];
ry(-1.6369275018423313) q[6];
rz(1.0273154156600532) q[6];
ry(0.27997093116203864) q[7];
rz(3.106293917108664) q[7];
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
ry(-1.0005272327518) q[0];
rz(-3.0043458481293857) q[0];
ry(1.928717007210654) q[1];
rz(2.34200247039655) q[1];
ry(-0.23005415545344302) q[2];
rz(-0.36822075647846564) q[2];
ry(-0.7765123662897935) q[3];
rz(2.596540523841339) q[3];
ry(-1.777068828302415) q[4];
rz(2.091456059196879) q[4];
ry(-2.43698571524123) q[5];
rz(1.4112436463284355) q[5];
ry(3.033712305187037) q[6];
rz(-1.4825137361706933) q[6];
ry(-2.464918706068077) q[7];
rz(0.7815901331815756) q[7];
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
ry(-1.9912230250585363) q[0];
rz(-3.047342680120245) q[0];
ry(2.892586698288822) q[1];
rz(1.9132111401740992) q[1];
ry(0.8819694221452454) q[2];
rz(0.939268477240761) q[2];
ry(2.2702680103431803) q[3];
rz(-3.0377975043748315) q[3];
ry(0.07736496826248093) q[4];
rz(-0.5325105251594513) q[4];
ry(-2.3383259282414905) q[5];
rz(-2.8112910735828716) q[5];
ry(2.2664950257818526) q[6];
rz(-0.37672430202659196) q[6];
ry(-2.1883039174990433) q[7];
rz(-0.597292644569916) q[7];
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
ry(-0.4452735779155187) q[0];
rz(-2.4579662867475944) q[0];
ry(-1.799239136779556) q[1];
rz(-0.03380912978455302) q[1];
ry(-2.7892886917311404) q[2];
rz(-2.6861535045056524) q[2];
ry(2.9565926768027713) q[3];
rz(1.4494484580295481) q[3];
ry(-1.4912896125599338) q[4];
rz(0.06970806304381943) q[4];
ry(-1.3361496656249554) q[5];
rz(0.4128349935925612) q[5];
ry(-2.6530013822646876) q[6];
rz(0.2285728985689435) q[6];
ry(-1.1735363524548754) q[7];
rz(0.16778851831989705) q[7];
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
ry(2.0002886464732494) q[0];
rz(0.12713324605157458) q[0];
ry(1.505680228045132) q[1];
rz(-2.52665756645462) q[1];
ry(0.7955925906279999) q[2];
rz(2.7033194855286045) q[2];
ry(0.5999217093855425) q[3];
rz(-1.5698299046796729) q[3];
ry(0.6801494709347625) q[4];
rz(0.4455827644499769) q[4];
ry(-2.7640530845977183) q[5];
rz(-1.2622132565118749) q[5];
ry(0.7336660563870865) q[6];
rz(-2.933930664042141) q[6];
ry(2.1170533820412905) q[7];
rz(-2.790742836307377) q[7];
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
ry(-0.2960485145022984) q[0];
rz(3.1016383950783255) q[0];
ry(1.9413501895733418) q[1];
rz(2.8302989585679574) q[1];
ry(1.1538805383925042) q[2];
rz(1.1455445297289817) q[2];
ry(1.790501013341144) q[3];
rz(-0.5323372968472996) q[3];
ry(0.9711654885334885) q[4];
rz(-1.0146679076260448) q[4];
ry(-1.0562189552381076) q[5];
rz(-0.8426223294059181) q[5];
ry(-1.9957861412491207) q[6];
rz(2.6491123935867296) q[6];
ry(-0.510315960838022) q[7];
rz(-0.10556737718645605) q[7];
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
ry(-2.86731215007022) q[0];
rz(-1.7874877431187377) q[0];
ry(1.6781460846004235) q[1];
rz(0.45990368562787876) q[1];
ry(-0.7911852615368318) q[2];
rz(0.25631367551914774) q[2];
ry(0.2973178759406725) q[3];
rz(3.066159846614952) q[3];
ry(2.6780210865850647) q[4];
rz(-0.9723057100078806) q[4];
ry(0.21613389662197893) q[5];
rz(2.1782283852074156) q[5];
ry(-0.39154786155694415) q[6];
rz(-1.1091191553780282) q[6];
ry(-1.3080511438507045) q[7];
rz(-1.6287607821865355) q[7];
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
ry(-0.15978691531973407) q[0];
rz(-2.1213433087438185) q[0];
ry(-1.8303819407137456) q[1];
rz(-1.3557586650509326) q[1];
ry(0.11658578023941056) q[2];
rz(-3.015725824577578) q[2];
ry(-2.7084593587036587) q[3];
rz(1.857650354472102) q[3];
ry(-1.6537406340262537) q[4];
rz(2.5217319727402066) q[4];
ry(-2.850778582944014) q[5];
rz(-0.6777714782459965) q[5];
ry(-1.7663480631736088) q[6];
rz(-0.8774393506453406) q[6];
ry(-0.6616549601182368) q[7];
rz(1.6264003806433085) q[7];
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
ry(2.4093991034286613) q[0];
rz(0.43803172116719746) q[0];
ry(2.1898066776872733) q[1];
rz(-3.081542961242493) q[1];
ry(-1.8301417212260427) q[2];
rz(1.0595807400534945) q[2];
ry(1.8132044793186874) q[3];
rz(-0.6991484181008403) q[3];
ry(-0.5518946287504057) q[4];
rz(-0.8811197782607785) q[4];
ry(0.6836949735454283) q[5];
rz(-3.0962288151144564) q[5];
ry(-1.8089234371300726) q[6];
rz(2.4522161321339024) q[6];
ry(0.477853440556865) q[7];
rz(-1.4249420626073812) q[7];
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
ry(0.5788457230594135) q[0];
rz(-1.3675779770622998) q[0];
ry(0.11279007908408645) q[1];
rz(-1.8350599649546595) q[1];
ry(-2.3946343551705342) q[2];
rz(-2.42782230096325) q[2];
ry(2.059721964909847) q[3];
rz(-1.8852397463660147) q[3];
ry(0.33279308647143285) q[4];
rz(-1.8267121788912315) q[4];
ry(2.565710418376895) q[5];
rz(0.23434182479904517) q[5];
ry(-2.844556664851227) q[6];
rz(-1.2305457620155633) q[6];
ry(1.7585469168971124) q[7];
rz(-0.8424653593238443) q[7];
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
ry(2.673577400387714) q[0];
rz(2.0893933417782415) q[0];
ry(-1.268828163648922) q[1];
rz(-2.0939937164024) q[1];
ry(2.5928739456660357) q[2];
rz(1.671350212548153) q[2];
ry(1.745288215412355) q[3];
rz(0.08834947133622399) q[3];
ry(1.2329612036480597) q[4];
rz(1.7864224473116228) q[4];
ry(1.1716855659829977) q[5];
rz(2.8785209232742806) q[5];
ry(0.659875656149598) q[6];
rz(1.5121351731214872) q[6];
ry(-1.152154007279691) q[7];
rz(2.281941223945391) q[7];
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
ry(-1.4053083917641622) q[0];
rz(-0.38679595618223495) q[0];
ry(2.5243818575111034) q[1];
rz(1.3472767692772267) q[1];
ry(1.2453147438199859) q[2];
rz(-2.090655393732738) q[2];
ry(-1.8791300745842099) q[3];
rz(-3.08281301332976) q[3];
ry(-2.93045621675871) q[4];
rz(1.3664979771923695) q[4];
ry(1.9246034395243092) q[5];
rz(-0.9225904854575253) q[5];
ry(-2.4880940056124823) q[6];
rz(2.977472566630717) q[6];
ry(0.39689441568074124) q[7];
rz(-2.877090609659584) q[7];
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
ry(2.513712147062695) q[0];
rz(1.1340250399250618) q[0];
ry(0.20951249913600112) q[1];
rz(-2.2088023009222635) q[1];
ry(2.637493856153694) q[2];
rz(0.7517316743712836) q[2];
ry(0.9297919152721389) q[3];
rz(1.7898027548943105) q[3];
ry(0.9421281658311902) q[4];
rz(-2.240031879711643) q[4];
ry(-1.9439960235664728) q[5];
rz(-1.5324951998354697) q[5];
ry(-0.12558360217428088) q[6];
rz(2.9831096394202077) q[6];
ry(2.0615102102474037) q[7];
rz(-2.0711378433513867) q[7];
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
ry(2.1580454073181614) q[0];
rz(2.148733740020343) q[0];
ry(-0.6860275075488619) q[1];
rz(0.43498929026044036) q[1];
ry(-1.1742123600437868) q[2];
rz(3.1298542938871945) q[2];
ry(0.10051362049305279) q[3];
rz(-2.7449624093525835) q[3];
ry(-0.4307668909715948) q[4];
rz(1.9488114980620148) q[4];
ry(-0.43261643251837967) q[5];
rz(0.21141095705575985) q[5];
ry(-1.5145326372753598) q[6];
rz(0.5391152814661524) q[6];
ry(-0.8507048608374319) q[7];
rz(-0.01926198425791092) q[7];
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
ry(-2.4742430488764016) q[0];
rz(-3.0279896890269917) q[0];
ry(1.0704006741013528) q[1];
rz(2.4401370557362534) q[1];
ry(-2.3573175406783586) q[2];
rz(-0.3249021142965845) q[2];
ry(1.131468323425195) q[3];
rz(-1.0567512982986225) q[3];
ry(-0.8740219393970067) q[4];
rz(-1.6889749533696252) q[4];
ry(-0.9231304880149614) q[5];
rz(-0.5371347511719944) q[5];
ry(-2.5021888432810253) q[6];
rz(0.5332419983436388) q[6];
ry(-1.8950460640881561) q[7];
rz(-0.34295840906641073) q[7];
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
ry(0.7121375897848924) q[0];
rz(-0.3265725645038366) q[0];
ry(-0.5969003476268402) q[1];
rz(0.32742908234141677) q[1];
ry(1.1876348195760762) q[2];
rz(-1.3990323615549727) q[2];
ry(-1.5423404858515775) q[3];
rz(-1.4259256098072517) q[3];
ry(2.927265979735791) q[4];
rz(-2.489379231042903) q[4];
ry(1.6977661164478468) q[5];
rz(1.2909511817732762) q[5];
ry(0.4524437552139687) q[6];
rz(-1.336882798699735) q[6];
ry(3.0631615107786185) q[7];
rz(0.8334622253388231) q[7];
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
ry(0.20146157935956222) q[0];
rz(-1.4919696888829665) q[0];
ry(-0.28459085966396014) q[1];
rz(2.3272576567303576) q[1];
ry(2.8508548564414236) q[2];
rz(1.1151035132446525) q[2];
ry(-0.7398633604870799) q[3];
rz(-0.7375954673766998) q[3];
ry(-1.6480011369613585) q[4];
rz(-1.6563397258663404) q[4];
ry(0.1950959578253054) q[5];
rz(1.4237025956498708) q[5];
ry(-1.358680861828005) q[6];
rz(-3.1074683553146096) q[6];
ry(2.8949716199019155) q[7];
rz(2.236334270866659) q[7];
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
ry(1.090652345618988) q[0];
rz(-3.1028962567623886) q[0];
ry(-2.164243497777446) q[1];
rz(-0.7201971675222664) q[1];
ry(2.6551468931909046) q[2];
rz(2.3963463158934775) q[2];
ry(2.608442639530746) q[3];
rz(0.7706673079206896) q[3];
ry(-2.1796942205366765) q[4];
rz(2.542672974637702) q[4];
ry(-1.8866955693846084) q[5];
rz(0.3606930742357541) q[5];
ry(2.0411256044965653) q[6];
rz(-2.0258833010784656) q[6];
ry(0.5886069364522859) q[7];
rz(-2.7268093485586484) q[7];
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
ry(1.1639157191295337) q[0];
rz(2.985475170263234) q[0];
ry(-2.798293302558049) q[1];
rz(-2.3723846190189852) q[1];
ry(2.8958794660628993) q[2];
rz(0.7705267363827168) q[2];
ry(1.954735485867132) q[3];
rz(-1.7676137295118726) q[3];
ry(1.5904561811159326) q[4];
rz(-1.47586176887731) q[4];
ry(2.87909148256583) q[5];
rz(1.8790286622855035) q[5];
ry(1.1464982320890174) q[6];
rz(0.3153539407343837) q[6];
ry(-0.6575551077009942) q[7];
rz(1.3146920842556995) q[7];
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
ry(-1.4140186679548084) q[0];
rz(-1.857660970661428) q[0];
ry(1.7435772203833393) q[1];
rz(1.429714219013829) q[1];
ry(0.7123374549495178) q[2];
rz(-2.403036733769379) q[2];
ry(2.019670350625674) q[3];
rz(-2.6731698647730884) q[3];
ry(-2.4723399070193004) q[4];
rz(-0.6588316297855835) q[4];
ry(-2.8760123648445313) q[5];
rz(2.8370487045648654) q[5];
ry(1.790350083972668) q[6];
rz(3.066291712292904) q[6];
ry(-2.772989862590917) q[7];
rz(0.70201995683617) q[7];
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
ry(-2.6738577301384954) q[0];
rz(-2.99246821148593) q[0];
ry(1.727074012927936) q[1];
rz(-0.44550027451637625) q[1];
ry(0.9674751268852573) q[2];
rz(-3.1082356210538333) q[2];
ry(2.185838791330389) q[3];
rz(-2.7343548070291273) q[3];
ry(0.8939258480424446) q[4];
rz(2.4217364694505368) q[4];
ry(1.6978704302994299) q[5];
rz(-2.6199032126836923) q[5];
ry(-2.666358951900899) q[6];
rz(-0.7364991631925708) q[6];
ry(1.0937402058307102) q[7];
rz(-0.6794940715323682) q[7];
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
ry(1.856570443260237) q[0];
rz(-0.20359129944283083) q[0];
ry(-1.5912603773767657) q[1];
rz(0.4231978777779072) q[1];
ry(3.0013340896966563) q[2];
rz(0.6624434062560463) q[2];
ry(2.3823948788902904) q[3];
rz(1.9842396859174858) q[3];
ry(1.3449958593924505) q[4];
rz(0.6964749649233423) q[4];
ry(-0.8880405389988777) q[5];
rz(-1.598262309019018) q[5];
ry(2.1092911922496835) q[6];
rz(-2.9249545604823837) q[6];
ry(2.941426158806038) q[7];
rz(2.6528134201790405) q[7];