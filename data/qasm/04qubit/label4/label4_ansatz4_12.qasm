OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.845716160166221) q[0];
rz(-1.8234169986881215) q[0];
ry(-1.6853835551204694) q[1];
rz(-2.31829228749411) q[1];
ry(0.630501766233506) q[2];
rz(1.4606047436117118) q[2];
ry(-1.6902659001251872) q[3];
rz(-1.8197621398050634) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.8151153347943572) q[0];
rz(-1.7369903019208872) q[0];
ry(2.3186339635437476) q[1];
rz(0.6125021395345991) q[1];
ry(1.960601502880703) q[2];
rz(2.1719837873888608) q[2];
ry(1.8556995076250224) q[3];
rz(-2.8076486883008887) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.381678691066083) q[0];
rz(3.1196829570825004) q[0];
ry(-0.9579923917578821) q[1];
rz(-2.116787751130175) q[1];
ry(-0.03838287322500696) q[2];
rz(3.097976703369536) q[2];
ry(-2.1948852215791503) q[3];
rz(-2.5813128343767198) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.3464057037420911) q[0];
rz(1.3774987147980415) q[0];
ry(0.09592445410572203) q[1];
rz(-1.2901567393364295) q[1];
ry(-1.4557641416393166) q[2];
rz(1.3169721611323721) q[2];
ry(3.015731949377528) q[3];
rz(2.1722133693635657) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.15375017020022508) q[0];
rz(-1.5671385066340129) q[0];
ry(-0.23823522142510267) q[1];
rz(1.31022852272075) q[1];
ry(0.4950130482691266) q[2];
rz(-1.3710369585175464) q[2];
ry(2.2133959539963057) q[3];
rz(2.77804593859134) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.4401074734943218) q[0];
rz(-1.4926342411697266) q[0];
ry(-0.5259055154585212) q[1];
rz(-3.0694192490854286) q[1];
ry(-1.0708015523978736) q[2];
rz(1.697182908524669) q[2];
ry(0.5257943701066284) q[3];
rz(3.0009949226522634) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.523966388851995) q[0];
rz(-1.263994695874326) q[0];
ry(-2.6303563525810887) q[1];
rz(-1.6834585738106103) q[1];
ry(0.6697180022407182) q[2];
rz(3.0135768046030944) q[2];
ry(1.0726428683155698) q[3];
rz(-1.7742766329416275) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.8703381301602888) q[0];
rz(0.41742346210223946) q[0];
ry(-2.111158091754997) q[1];
rz(2.430322387185727) q[1];
ry(1.896354094904105) q[2];
rz(0.8204263684730231) q[2];
ry(-2.5684994859620045) q[3];
rz(2.691915316630165) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.06554442219891) q[0];
rz(3.0541204485391207) q[0];
ry(-2.0127044709546738) q[1];
rz(3.0592660426773364) q[1];
ry(0.7953886033124421) q[2];
rz(-1.264081789273791) q[2];
ry(-2.4064861605905365) q[3];
rz(0.2871237555298423) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.5801246101815494) q[0];
rz(1.114891054260422) q[0];
ry(-2.0223707345457775) q[1];
rz(0.1385933007907116) q[1];
ry(-2.3962024025040023) q[2];
rz(0.4166881787194674) q[2];
ry(-0.24434135424896325) q[3];
rz(-1.0690827235975462) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.243181991773382) q[0];
rz(-1.9524227882290965) q[0];
ry(-2.3203547434924587) q[1];
rz(1.6202120043820398) q[1];
ry(0.19652902869919764) q[2];
rz(2.710308207532812) q[2];
ry(-0.09438683896197313) q[3];
rz(0.10931045511309812) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.8440947428099103) q[0];
rz(0.9399487192769183) q[0];
ry(-1.8097867118519002) q[1];
rz(-2.9664307405075987) q[1];
ry(0.7942351824041323) q[2];
rz(-1.9316106800194124) q[2];
ry(-1.754023571080177) q[3];
rz(-0.23589911918223405) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.661440294857808) q[0];
rz(-1.3731004521306998) q[0];
ry(1.5850409426982346) q[1];
rz(2.5673757825855312) q[1];
ry(2.707768657267942) q[2];
rz(1.2725548278410788) q[2];
ry(1.8525341072586032) q[3];
rz(-1.2011350650737918) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3090512511729917) q[0];
rz(-0.9804872770169784) q[0];
ry(2.4322083135089505) q[1];
rz(-2.058583737502153) q[1];
ry(2.653455511260222) q[2];
rz(1.1354688200536553) q[2];
ry(2.6211252071077804) q[3];
rz(-1.9765003650289277) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4254987699875827) q[0];
rz(2.2081477889188452) q[0];
ry(-1.2263634498785656) q[1];
rz(-0.286144998764216) q[1];
ry(0.7583380390422272) q[2];
rz(-2.893267246583821) q[2];
ry(0.35319577635028665) q[3];
rz(2.8993520124455685) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9602588092804445) q[0];
rz(-0.48711578471818984) q[0];
ry(1.0016218944054873) q[1];
rz(1.0464108570587756) q[1];
ry(-1.6075865217978702) q[2];
rz(0.3179433597151373) q[2];
ry(-2.1789036272837836) q[3];
rz(1.4290280706907448) q[3];