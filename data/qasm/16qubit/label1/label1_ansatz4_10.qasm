OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.6295306638160975) q[0];
rz(1.841663660672942) q[0];
ry(-0.11767880343458614) q[1];
rz(-2.219172350685865) q[1];
ry(-3.1389611544479417) q[2];
rz(-0.21756998503042893) q[2];
ry(3.1341711079805816) q[3];
rz(0.2064405974493937) q[3];
ry(0.4289172567123467) q[4];
rz(1.7906394691850087) q[4];
ry(2.1162582467938247) q[5];
rz(2.3723073141494133) q[5];
ry(0.1828368312572719) q[6];
rz(2.37941452716907) q[6];
ry(0.25453705026451734) q[7];
rz(1.8460163131667977) q[7];
ry(-0.006998678986212715) q[8];
rz(-1.3081800767458023) q[8];
ry(-0.013570370151919541) q[9];
rz(1.0888491926872863) q[9];
ry(-3.096391575977375) q[10];
rz(1.21676431071677) q[10];
ry(0.011150655135224952) q[11];
rz(-1.2294315777216644) q[11];
ry(1.5802567620691093) q[12];
rz(-0.4638962571118418) q[12];
ry(1.5688875712340904) q[13];
rz(0.09566092211822844) q[13];
ry(0.3319285193530981) q[14];
rz(2.916971185261762) q[14];
ry(3.116494903122636) q[15];
rz(1.5285229071087212) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.6271335621812169) q[0];
rz(-1.7987952172172132) q[0];
ry(-2.407358873747909) q[1];
rz(-0.3764003248811146) q[1];
ry(-3.1362167314011704) q[2];
rz(1.9802490598121592) q[2];
ry(-0.006197165928666769) q[3];
rz(2.4341068115499116) q[3];
ry(-2.29202961342806) q[4];
rz(2.0775536628435862) q[4];
ry(-1.1612539120753653) q[5];
rz(1.3999007392112837) q[5];
ry(1.498681249708003) q[6];
rz(0.35498042388608564) q[6];
ry(2.2757142078949144) q[7];
rz(2.4888187552472023) q[7];
ry(3.1293235136166158) q[8];
rz(1.3813418440815615) q[8];
ry(-3.1412283610476988) q[9];
rz(0.7179649891358179) q[9];
ry(-1.7259339218026035) q[10];
rz(-2.798942965054304) q[10];
ry(-0.6945381863801856) q[11];
rz(0.789779257133743) q[11];
ry(-2.9529597360612456) q[12];
rz(-0.38599705178109306) q[12];
ry(-1.9390503985829577) q[13];
rz(-3.001473550970694) q[13];
ry(-2.526471686580757) q[14];
rz(-1.634144480423976) q[14];
ry(0.6642222708334211) q[15];
rz(2.899466931732252) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.1803142720824864) q[0];
rz(-2.2648202631919467) q[0];
ry(0.3416067649849497) q[1];
rz(2.421055249973483) q[1];
ry(3.134210290041394) q[2];
rz(1.801677434799914) q[2];
ry(0.023464582483502028) q[3];
rz(1.9925532733972668) q[3];
ry(0.7620776197788249) q[4];
rz(0.9879046922387245) q[4];
ry(-2.3120989000601444) q[5];
rz(-0.356383809130563) q[5];
ry(2.3673696564379445) q[6];
rz(-2.4232658503115068) q[6];
ry(-0.17558468290788376) q[7];
rz(0.23929280446982126) q[7];
ry(-2.3955464044551564) q[8];
rz(0.6911719323712546) q[8];
ry(1.9110164486623484) q[9];
rz(0.7319178774070773) q[9];
ry(1.839679557189693) q[10];
rz(-1.4830288551200208) q[10];
ry(2.902694058206166) q[11];
rz(2.3147555567209896) q[11];
ry(-1.665566334992795) q[12];
rz(0.6671896812910214) q[12];
ry(-1.5545250057152644) q[13];
rz(-2.088352217108302) q[13];
ry(1.7752620703372293) q[14];
rz(-2.056105450356503) q[14];
ry(0.6889663469330892) q[15];
rz(-2.845794580367519) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.569807930611181) q[0];
rz(-0.9580698000444121) q[0];
ry(2.9834544625378316) q[1];
rz(2.8502184666083155) q[1];
ry(-2.8889007901982504) q[2];
rz(0.9820392735974769) q[2];
ry(0.971781205918346) q[3];
rz(-0.44794452286243214) q[3];
ry(-1.3807534166528184) q[4];
rz(-0.06060030969792671) q[4];
ry(2.066952696627201) q[5];
rz(-0.9807524308432498) q[5];
ry(-0.012316701888490654) q[6];
rz(-2.831089233006734) q[6];
ry(0.012869085115529035) q[7];
rz(-1.8378181897750538) q[7];
ry(-3.0919656223198193) q[8];
rz(0.7662815870764703) q[8];
ry(3.0861930872553476) q[9];
rz(1.5091787667679804) q[9];
ry(2.719603657201336) q[10];
rz(-0.05588940461087155) q[10];
ry(1.0928626644150516) q[11];
rz(-0.12171552502049282) q[11];
ry(-3.046942982854472) q[12];
rz(0.47010850799417786) q[12];
ry(0.215934834633072) q[13];
rz(-0.6207541582228054) q[13];
ry(0.15452253404920846) q[14];
rz(2.5860809443884687) q[14];
ry(-0.8107531211891139) q[15];
rz(2.7268173331056684) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.947836683340273) q[0];
rz(2.0583013987289265) q[0];
ry(-0.5442683443209368) q[1];
rz(-0.32570912235483357) q[1];
ry(0.09044919188187597) q[2];
rz(-2.305379865978256) q[2];
ry(-0.4688984594283747) q[3];
rz(0.8708610142867721) q[3];
ry(0.8885286928069939) q[4];
rz(3.0134788877904755) q[4];
ry(-0.07599258100229607) q[5];
rz(-1.1284374000482658) q[5];
ry(1.8306629059767088) q[6];
rz(-2.664441488682815) q[6];
ry(1.3815290436719874) q[7];
rz(0.32190849735226124) q[7];
ry(3.075176992626193) q[8];
rz(-0.02859906252106887) q[8];
ry(-0.03775415533197535) q[9];
rz(-1.0261862445961196) q[9];
ry(-3.005158636919687) q[10];
rz(3.015619396648314) q[10];
ry(-3.140509805943079) q[11];
rz(2.608391591311735) q[11];
ry(1.4309521014398383) q[12];
rz(1.997855621960734) q[12];
ry(1.6898793660435676) q[13];
rz(-0.7438155204842101) q[13];
ry(-1.553382356911694) q[14];
rz(1.5947370134937477) q[14];
ry(1.6728186643497835) q[15];
rz(0.04162150460446146) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.202429785818068) q[0];
rz(2.7189602213765967) q[0];
ry(-0.5247084734723013) q[1];
rz(-2.913771233782568) q[1];
ry(-1.3313862329524433) q[2];
rz(-2.2479727136056873) q[2];
ry(-3.1224735717630456) q[3];
rz(0.25921283282933505) q[3];
ry(-3.0678793405124196) q[4];
rz(2.2229604150172273) q[4];
ry(-2.313303234628101) q[5];
rz(1.6508801437629215) q[5];
ry(1.5614078744603832) q[6];
rz(-3.1368570936390956) q[6];
ry(1.570367428374374) q[7];
rz(9.649303987746549e-05) q[7];
ry(3.0122447363395537) q[8];
rz(1.6053773459998817) q[8];
ry(-3.095001779393414) q[9];
rz(-1.7298214866874622) q[9];
ry(0.3337017327710835) q[10];
rz(-1.5429326885421326) q[10];
ry(-0.15147643245592501) q[11];
rz(-1.089040502988838) q[11];
ry(-3.0047581902096114) q[12];
rz(2.712410229726057) q[12];
ry(2.5282042343711373) q[13];
rz(-2.5121470979727305) q[13];
ry(1.5912893676120525) q[14];
rz(-2.3162873408225253) q[14];
ry(-1.535451198815356) q[15];
rz(1.5468332617177696) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.2898946934014832) q[0];
rz(-0.060302856576228595) q[0];
ry(1.6059007656954485) q[1];
rz(0.14156527167494826) q[1];
ry(3.138351365872194) q[2];
rz(-2.030768545355298) q[2];
ry(-0.0010599018289036977) q[3];
rz(-1.1855022988538442) q[3];
ry(-3.1334080924802) q[4];
rz(-1.5336741398520752) q[4];
ry(-0.009086750980725797) q[5];
rz(-0.7042385007637999) q[5];
ry(1.5737091191668675) q[6];
rz(2.9593367675653424) q[6];
ry(-1.5821144843042925) q[7];
rz(0.032778262714508344) q[7];
ry(-1.571959446143392) q[8];
rz(1.6093097451037481) q[8];
ry(1.590717963630761) q[9];
rz(-1.4222354519039628) q[9];
ry(-0.010203511431881473) q[10];
rz(1.3458230719235185) q[10];
ry(-3.1303465056075845) q[11];
rz(-2.3903257720917597) q[11];
ry(3.031415754959171) q[12];
rz(2.9550600438935337) q[12];
ry(0.6480629227994018) q[13];
rz(2.3897609877672665) q[13];
ry(-1.5923730947144028) q[14];
rz(-1.5549478236824983) q[14];
ry(1.5605810405304172) q[15];
rz(1.9232575163761374) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.8239907718778463) q[0];
rz(1.2637107428708902) q[0];
ry(-2.5420072208167097) q[1];
rz(0.30684082500467635) q[1];
ry(1.9763871503343164) q[2];
rz(0.5759482257230433) q[2];
ry(-0.31301881540695586) q[3];
rz(1.4340540747712285) q[3];
ry(0.6475352312597253) q[4];
rz(-1.9188927906543691) q[4];
ry(1.1683885827806921) q[5];
rz(0.9301815126136459) q[5];
ry(3.0196696749787004) q[6];
rz(-2.1843386781110383) q[6];
ry(-0.2905878602506781) q[7];
rz(0.29026934469901994) q[7];
ry(-2.264140671812226) q[8];
rz(0.009211784883001102) q[8];
ry(-0.541927501103402) q[9];
rz(-0.2663693979302897) q[9];
ry(-3.0605523069192824) q[10];
rz(-0.213952551255491) q[10];
ry(-0.02131398207923656) q[11];
rz(-0.6488803564973196) q[11];
ry(2.734015006165534) q[12];
rz(-2.6340076579398555) q[12];
ry(-1.1884262635492464) q[13];
rz(-2.0223127732024206) q[13];
ry(1.6008750931464446) q[14];
rz(0.27222728145291164) q[14];
ry(-1.593474695477913) q[15];
rz(0.0019465445341688725) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.510826566079063) q[0];
rz(-0.14761159788930925) q[0];
ry(2.5909591211660365) q[1];
rz(-2.835425595977582) q[1];
ry(0.009281699889939432) q[2];
rz(-0.4681014219727092) q[2];
ry(-0.012405491417386116) q[3];
rz(1.3896988300575022) q[3];
ry(0.0010017114060945078) q[4];
rz(-0.49782958869166904) q[4];
ry(-3.1408272771650063) q[5];
rz(-2.7625324491057586) q[5];
ry(3.1249392780419942) q[6];
rz(2.4241148166149893) q[6];
ry(-3.1298575418393066) q[7];
rz(2.710546985838029) q[7];
ry(1.7996921050856503) q[8];
rz(-0.613544217532672) q[8];
ry(1.426947683011548) q[9];
rz(0.14845655345842346) q[9];
ry(-1.455941526790402) q[10];
rz(1.4526509756198696) q[10];
ry(1.6017278957643035) q[11];
rz(0.18325531766137448) q[11];
ry(-1.9173956655966677) q[12];
rz(1.9721952507536002) q[12];
ry(2.2415265153144723) q[13];
rz(1.137228393401638) q[13];
ry(1.5570661999860036) q[14];
rz(-1.5865260101093388) q[14];
ry(1.6643146708579895) q[15];
rz(0.06543496535922434) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.3588873717657104) q[0];
rz(-2.7555320276925865) q[0];
ry(-1.993928893843111) q[1];
rz(-1.4963983405922372) q[1];
ry(1.6828877667358801) q[2];
rz(2.16154419585662) q[2];
ry(3.1317764355026556) q[3];
rz(-2.2319331001319433) q[3];
ry(-0.40328268541445217) q[4];
rz(-0.6361879781430733) q[4];
ry(-1.0505474371116152) q[5];
rz(-0.023800215415944233) q[5];
ry(0.07993584639617848) q[6];
rz(1.749933625891357) q[6];
ry(-2.6366802979603507) q[7];
rz(2.523653489308615) q[7];
ry(-3.101994231642089) q[8];
rz(2.561070456186586) q[8];
ry(-0.09225456384330318) q[9];
rz(-1.8746870693943052) q[9];
ry(-0.03409986329915053) q[10];
rz(-3.0041324804500404) q[10];
ry(3.1413531135629604) q[11];
rz(1.4813844911599476) q[11];
ry(3.0592024060841676) q[12];
rz(-0.7318263314048252) q[12];
ry(0.21468328513335955) q[13];
rz(0.28936051025813647) q[13];
ry(-0.8095837339954413) q[14];
rz(0.06643838168575034) q[14];
ry(1.6315461115636964) q[15];
rz(0.06373079492336318) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.185727086750081) q[0];
rz(-2.8871974941051888) q[0];
ry(0.0404588202234617) q[1];
rz(2.958205489391171) q[1];
ry(1.6141752322265885) q[2];
rz(-1.556362858111101) q[2];
ry(-1.5370994135770355) q[3];
rz(-2.8348295668149293) q[3];
ry(0.005452000471856344) q[4];
rz(-0.9779874124369936) q[4];
ry(-3.1371327925831185) q[5];
rz(-2.0471975120183985) q[5];
ry(-3.1376859686904623) q[6];
rz(1.3392736641923007) q[6];
ry(3.12537860556692) q[7];
rz(1.8171507162672709) q[7];
ry(2.5535367039707997) q[8];
rz(-2.1791807162140806) q[8];
ry(-0.05855368886645174) q[9];
rz(-1.5684668557258217) q[9];
ry(-1.5816137092864226) q[10];
rz(-3.060161607139131) q[10];
ry(-1.5075122515030204) q[11];
rz(-1.7891719075411936) q[11];
ry(0.40811212999404756) q[12];
rz(-0.39148435929443803) q[12];
ry(-1.2814653085898753) q[13];
rz(0.13088727271272746) q[13];
ry(-0.010852414484992854) q[14];
rz(3.079855074004223) q[14];
ry(0.30801081309155) q[15];
rz(3.1167911456287007) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.18971175270284935) q[0];
rz(2.5411491058901703) q[0];
ry(3.087397192503076) q[1];
rz(-1.2069403081455672) q[1];
ry(-0.4551971782213702) q[2];
rz(1.5683625843754296) q[2];
ry(0.003944111077410106) q[3];
rz(2.8341639979301108) q[3];
ry(-0.09237784336836173) q[4];
rz(-1.9693237088268747) q[4];
ry(3.1240817509774836) q[5];
rz(2.6998658532468642) q[5];
ry(-3.0451745490015605) q[6];
rz(-1.469997234994052) q[6];
ry(0.4237621339840505) q[7];
rz(1.9196435892437917) q[7];
ry(0.02675806083079515) q[8];
rz(0.6525305258389825) q[8];
ry(0.060428615411513516) q[9];
rz(1.7302168453271012) q[9];
ry(-3.1392801249617768) q[10];
rz(1.547116269831557) q[10];
ry(3.123384537300242) q[11];
rz(2.872853742623837) q[11];
ry(0.019694648512626804) q[12];
rz(0.01126964462254726) q[12];
ry(2.9470567530775003) q[13];
rz(1.5890303970325945) q[13];
ry(1.518995799151647) q[14];
rz(-1.9218120979808062) q[14];
ry(0.16991095170761739) q[15];
rz(-3.025955326798396) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.5090710478258398) q[0];
rz(-1.5059217658223512) q[0];
ry(-1.10406582557527) q[1];
rz(-0.05748178309691799) q[1];
ry(-1.6433180737367596) q[2];
rz(-1.6749704431759904) q[2];
ry(1.4930289879698337) q[3];
rz(-2.9301266202643377) q[3];
ry(3.1390797262299652) q[4];
rz(-1.2958420927132948) q[4];
ry(-3.133267088231458) q[5];
rz(2.142420127355732) q[5];
ry(-3.111660774858715) q[6];
rz(0.2524346744592982) q[6];
ry(3.1331047570900656) q[7];
rz(0.4710280853896448) q[7];
ry(1.5764423895176396) q[8];
rz(0.7206148317410008) q[8];
ry(-1.7559571776939513) q[9];
rz(0.10150007585262931) q[9];
ry(-1.5570929880030737) q[10];
rz(1.6338082387824961) q[10];
ry(-1.8830946525744707) q[11];
rz(-0.23233666305733977) q[11];
ry(1.6220546467325356) q[12];
rz(0.5782996124906646) q[12];
ry(-1.6999817762475786) q[13];
rz(2.724463534103925) q[13];
ry(3.097819565035526) q[14];
rz(-1.8129966165823848) q[14];
ry(0.18637120477889813) q[15];
rz(0.2282729680083939) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.2998253802369968) q[0];
rz(1.4827328598641243) q[0];
ry(0.09359782118419169) q[1];
rz(-0.11989868398111812) q[1];
ry(2.950594105745081) q[2];
rz(0.3489580156544374) q[2];
ry(-0.8035010131011653) q[3];
rz(1.3560173495951444) q[3];
ry(1.5645462447562242) q[4];
rz(1.3137164043676988) q[4];
ry(-1.355006569266774) q[5];
rz(0.7771300361044462) q[5];
ry(1.1835840714953711) q[6];
rz(-0.31068417954321603) q[6];
ry(0.8839117997630757) q[7];
rz(-0.17630020009043348) q[7];
ry(-1.3038040124048127) q[8];
rz(2.855781716201155) q[8];
ry(2.711806937691279) q[9];
rz(2.9476865274894246) q[9];
ry(0.28418817512013916) q[10];
rz(-0.25735640654678527) q[10];
ry(-2.853811344898358) q[11];
rz(2.745829407140245) q[11];
ry(-1.5652887589741844) q[12];
rz(1.3816424463907728) q[12];
ry(1.6234121260331698) q[13];
rz(-1.8472053275280278) q[13];
ry(-0.7043838780743334) q[14];
rz(2.7738559765002915) q[14];
ry(3.002736390887847) q[15];
rz(0.08952929324626556) q[15];