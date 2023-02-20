OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.7514352250190557) q[0];
rz(1.2152861807575799) q[0];
ry(-2.787532033411168) q[1];
rz(2.2566396621088387) q[1];
ry(0.001173381875816837) q[2];
rz(-2.7180890847704258) q[2];
ry(-0.013590242115042448) q[3];
rz(0.948793640475868) q[3];
ry(1.6275184948251356) q[4];
rz(-2.1001439211724184) q[4];
ry(-2.6709169131946546) q[5];
rz(-0.6416786267002438) q[5];
ry(-2.9382217413749476) q[6];
rz(-3.1398492873569825) q[6];
ry(0.32449600336863804) q[7];
rz(2.2088503693980304) q[7];
ry(-1.5810483581691033) q[8];
rz(-3.0706400895085566) q[8];
ry(1.5023345707146059) q[9];
rz(-0.1774300950269776) q[9];
ry(1.7018832239489692) q[10];
rz(-1.750885419191377) q[10];
ry(-0.07662526349855803) q[11];
rz(2.6823408521576027) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.9749867402676378) q[0];
rz(-2.9217287584499583) q[0];
ry(1.1581136830898733) q[1];
rz(-2.8301206357337634) q[1];
ry(1.6995884410125308) q[2];
rz(3.068289642161721) q[2];
ry(-1.7222821094554668) q[3];
rz(-2.843398740255781) q[3];
ry(1.014840625257639) q[4];
rz(-2.2643347136654137) q[4];
ry(-2.921296426700589) q[5];
rz(-2.2351068914606826) q[5];
ry(-3.138708975917579) q[6];
rz(-0.8853544067051855) q[6];
ry(-0.00803942511736544) q[7];
rz(-1.7117265188567048) q[7];
ry(2.4646841598428173) q[8];
rz(-1.8487850108135149) q[8];
ry(-2.453422948976125) q[9];
rz(1.5023720957890425) q[9];
ry(-1.656542760203032) q[10];
rz(-2.552278239190315) q[10];
ry(-2.8669892933653314) q[11];
rz(1.0530851691221246) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.4857491582451416) q[0];
rz(-1.8144442949619526) q[0];
ry(-2.2401747310587545) q[1];
rz(-1.0757328862005355) q[1];
ry(0.11751062089166499) q[2];
rz(-1.2215123665292984) q[2];
ry(2.5934408192172276) q[3];
rz(-1.5246168174004886) q[3];
ry(0.5790332485620046) q[4];
rz(2.4550402476334803) q[4];
ry(-0.2648531865984369) q[5];
rz(1.223246272981292) q[5];
ry(-2.888842922992499) q[6];
rz(-2.700608827933206) q[6];
ry(-0.41769332863648334) q[7];
rz(-3.0256168412482127) q[7];
ry(0.2934618088101253) q[8];
rz(-1.2133306813019331) q[8];
ry(2.463544352108527) q[9];
rz(-2.371685982816057) q[9];
ry(2.143274235936879) q[10];
rz(2.0807101918369253) q[10];
ry(-1.6886218358145602) q[11];
rz(-2.076940188762751) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.08308618419235816) q[0];
rz(-1.9579510747075644) q[0];
ry(-0.03015739392506983) q[1];
rz(-0.354863850413043) q[1];
ry(-1.7102815888134186) q[2];
rz(-1.4224725712002504) q[2];
ry(-1.6091095776534174) q[3];
rz(0.18495836028434845) q[3];
ry(0.17767147858667814) q[4];
rz(-0.8079547847863384) q[4];
ry(-0.06509914531482153) q[5];
rz(-1.8135058821275474) q[5];
ry(-3.1339614732999608) q[6];
rz(1.6766427365376666) q[6];
ry(0.001185736945848781) q[7];
rz(-0.40935425155459176) q[7];
ry(-3.1250279148411724) q[8];
rz(0.06829838316417586) q[8];
ry(-3.1134569482286194) q[9];
rz(-1.4214872333904562) q[9];
ry(2.9257387529964425) q[10];
rz(-2.359643800754478) q[10];
ry(-2.633689616159492) q[11];
rz(1.4554401545856042) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.047318379381222) q[0];
rz(-0.13849009875622453) q[0];
ry(-3.000136748393792) q[1];
rz(2.6383367130173916) q[1];
ry(0.61715669505673) q[2];
rz(2.3544835271726847) q[2];
ry(3.03919366928457) q[3];
rz(1.7394992040999604) q[3];
ry(-0.12749153193291374) q[4];
rz(-2.194773521414761) q[4];
ry(-3.133802047615695) q[5];
rz(1.0844823125869743) q[5];
ry(-3.055067358498748) q[6];
rz(-1.1954179772146114) q[6];
ry(-0.7039725717254547) q[7];
rz(-1.2330608243116234) q[7];
ry(-2.3805588599829695) q[8];
rz(2.1016138605461956) q[8];
ry(-1.0147315474670904) q[9];
rz(0.775244441460082) q[9];
ry(-0.41391183561295275) q[10];
rz(0.6235916991685466) q[10];
ry(2.896120065530198) q[11];
rz(2.027160346609243) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.4011004729876735) q[0];
rz(2.504422659384333) q[0];
ry(-0.2024322928597515) q[1];
rz(-0.2320532843956098) q[1];
ry(0.036332321866010986) q[2];
rz(-0.9670726279231426) q[2];
ry(-0.006950591092542702) q[3];
rz(-1.6546580007274887) q[3];
ry(-1.2629197637125438) q[4];
rz(-1.6019502021656953) q[4];
ry(1.9420194743846337) q[5];
rz(0.6370890890347413) q[5];
ry(1.5845701443005158) q[6];
rz(0.2207598413215549) q[6];
ry(-1.7239118920882897) q[7];
rz(-1.8249293037611405) q[7];
ry(-0.009040057135618389) q[8];
rz(1.072691807196538) q[8];
ry(-0.00443902336849259) q[9];
rz(-1.4417880333979134) q[9];
ry(2.960451355969933) q[10];
rz(3.1221368010366533) q[10];
ry(-1.6474226516528478) q[11];
rz(-0.9833543941642904) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.2453034338925812) q[0];
rz(1.337469982214765) q[0];
ry(-0.13560106729749322) q[1];
rz(0.8381528439496013) q[1];
ry(3.1383542741589485) q[2];
rz(-1.6635355565339047) q[2];
ry(3.134208536509395) q[3];
rz(-1.6979979704211905) q[3];
ry(0.015310651428675062) q[4];
rz(0.2960634390374552) q[4];
ry(-3.138167308369382) q[5];
rz(1.7288912265468017) q[5];
ry(1.5809246457712174) q[6];
rz(3.015275310486449) q[6];
ry(0.572433658072443) q[7];
rz(2.50391115004441) q[7];
ry(0.04043681216860051) q[8];
rz(2.2240890277136227) q[8];
ry(3.129215594215304) q[9];
rz(-0.593780995522701) q[9];
ry(2.7216068826395445) q[10];
rz(-0.060455360949872666) q[10];
ry(-2.8613055692835507) q[11];
rz(-0.5731745631200212) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.221902487191497) q[0];
rz(1.9635925072102844) q[0];
ry(0.546666686402542) q[1];
rz(0.9990339799819231) q[1];
ry(-1.2672946729600687) q[2];
rz(-1.6694808542534263) q[2];
ry(-1.8953813374722879) q[3];
rz(1.6934811661767792) q[3];
ry(-1.8985957738282186) q[4];
rz(0.6406586173617894) q[4];
ry(0.9732994736428948) q[5];
rz(-1.2347156883727994) q[5];
ry(-0.31943637621325993) q[6];
rz(-1.8791407073900441) q[6];
ry(1.95387978750479) q[7];
rz(2.113700526841014) q[7];
ry(0.048334470323693424) q[8];
rz(-1.2358156551688815) q[8];
ry(-3.0957262665762695) q[9];
rz(-1.8800322664056435) q[9];
ry(-0.7984480774658725) q[10];
rz(-2.7445067907732352) q[10];
ry(-3.1307713694912556) q[11];
rz(-2.355072906912281) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.9646211706494014) q[0];
rz(0.35875882249269536) q[0];
ry(-3.131009615382517) q[1];
rz(1.4174458658564637) q[1];
ry(1.5740793297075808) q[2];
rz(-1.9307422382198787) q[2];
ry(1.557956383033277) q[3];
rz(-0.024078952731263797) q[3];
ry(-1.6977415943551222) q[4];
rz(-2.618203178607584) q[4];
ry(0.79761379707075) q[5];
rz(-2.2069469883988333) q[5];
ry(0.6256731133965605) q[6];
rz(-0.15120127452693755) q[6];
ry(1.2682410933441863) q[7];
rz(0.3770483811224681) q[7];
ry(3.1330185903452974) q[8];
rz(1.6834754927254219) q[8];
ry(-0.016763072077309) q[9];
rz(0.21194009395294272) q[9];
ry(-3.0170911473044453) q[10];
rz(1.6463675879312687) q[10];
ry(-2.510441210440822) q[11];
rz(-2.865443953640744) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(2.4959029774057755) q[0];
rz(1.5280374893258601) q[0];
ry(2.2444608987324512) q[1];
rz(-2.988330974355557) q[1];
ry(-0.00351321841902763) q[2];
rz(-0.44050853160196535) q[2];
ry(-2.5244922634500604) q[3];
rz(1.728546331401741) q[3];
ry(3.1131104157850547) q[4];
rz(3.0681942011845376) q[4];
ry(-0.07336574475567659) q[5];
rz(-2.626662202524686) q[5];
ry(-2.587602815387369) q[6];
rz(1.3809428107046848) q[6];
ry(-1.914485208706565) q[7];
rz(-0.15528965109133713) q[7];
ry(-1.774893557699916) q[8];
rz(-0.5042905196934448) q[8];
ry(-2.0005515741679556) q[9];
rz(-0.3465026571179459) q[9];
ry(0.8858162429258659) q[10];
rz(-0.8347716129963324) q[10];
ry(-0.5598757361880882) q[11];
rz(-1.3193273419460052) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.967896763539936) q[0];
rz(2.670760832003583) q[0];
ry(-1.3761926833962779) q[1];
rz(-1.8489418440927112) q[1];
ry(0.3544581678395282) q[2];
rz(1.3045026194270257) q[2];
ry(0.47178529894987176) q[3];
rz(0.15014976203744465) q[3];
ry(2.369479551597226) q[4];
rz(-1.7603826422271727) q[4];
ry(2.869453550876512) q[5];
rz(2.4180212293116052) q[5];
ry(3.141430636491482) q[6];
rz(-1.4280298122931954) q[6];
ry(0.05133349396696473) q[7];
rz(3.023111885860291) q[7];
ry(-1.6081719000514223) q[8];
rz(1.7116954913616198) q[8];
ry(-1.5391401324973275) q[9];
rz(1.6914630640276374) q[9];
ry(-2.5087928032617044) q[10];
rz(0.3751110742721904) q[10];
ry(2.5101705845347677) q[11];
rz(-1.9947359500407469) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.0036668563143153763) q[0];
rz(-1.6345990650090638) q[0];
ry(0.23646608990138862) q[1];
rz(1.1535362710417008) q[1];
ry(-3.1386518302419204) q[2];
rz(-2.033503408752016) q[2];
ry(-3.13970668393682) q[3];
rz(-2.577258838733441) q[3];
ry(-3.1169862637456878) q[4];
rz(-2.554354713425983) q[4];
ry(-0.022892950626724584) q[5];
rz(-2.2587921582156296) q[5];
ry(-2.9650510018185026) q[6];
rz(2.7354957776894024) q[6];
ry(0.31185422762170883) q[7];
rz(-0.18138296834323955) q[7];
ry(0.9896074805031657) q[8];
rz(1.1195732038338857) q[8];
ry(-2.3306712857045433) q[9];
rz(-1.4778044884577248) q[9];
ry(-1.9979258135151392) q[10];
rz(2.903853251793684) q[10];
ry(-2.7116747014830342) q[11];
rz(2.6377442052154048) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.000792571394978) q[0];
rz(-2.4713590152778675) q[0];
ry(0.08172707518174704) q[1];
rz(-2.639158519461848) q[1];
ry(1.2750109053601404) q[2];
rz(-2.377809805439229) q[2];
ry(-0.33757267180557815) q[3];
rz(-0.35554167193948055) q[3];
ry(-1.382576975915021) q[4];
rz(2.3026523285730454) q[4];
ry(-1.5886137018994813) q[5];
rz(-0.08052432844162195) q[5];
ry(1.5934810059173925) q[6];
rz(1.4713446422613359) q[6];
ry(1.6871491172275304) q[7];
rz(3.111816782361447) q[7];
ry(3.122237639637745) q[8];
rz(2.7427714346322873) q[8];
ry(-0.023837499976180787) q[9];
rz(0.0032543058041545336) q[9];
ry(-0.2953979950530621) q[10];
rz(-2.0255112626085716) q[10];
ry(-1.2449114506608847) q[11];
rz(1.140504315557211) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.5905117672937499) q[0];
rz(-0.4817899588704159) q[0];
ry(-0.5973064369923797) q[1];
rz(2.5992985459415685) q[1];
ry(-3.0982257993719293) q[2];
rz(3.1029897902229147) q[2];
ry(3.1380451957092546) q[3];
rz(-2.4363747074143887) q[3];
ry(-1.703390636723938) q[4];
rz(1.2821401791035973) q[4];
ry(2.262660496079981) q[5];
rz(1.705789853558537) q[5];
ry(-0.2742544015393378) q[6];
rz(2.9590422308292372) q[6];
ry(1.5796633600584293) q[7];
rz(-2.383532974813855) q[7];
ry(1.5261565314152985) q[8];
rz(-1.5645750925763344) q[8];
ry(3.091796765780591) q[9];
rz(1.5199191566198458) q[9];
ry(-1.73359786794935) q[10];
rz(-1.82168230351589) q[10];
ry(3.033237940390313) q[11];
rz(1.3699544234346224) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.8761485002741907) q[0];
rz(0.9782132273598592) q[0];
ry(2.807605437464001) q[1];
rz(2.2049482332961814) q[1];
ry(-0.012336668890314966) q[2];
rz(-0.29548580113733885) q[2];
ry(-0.005140407102072864) q[3];
rz(0.7795279885928942) q[3];
ry(-0.0032854401960635826) q[4];
rz(-2.784837702021025) q[4];
ry(-3.1157348347940936) q[5];
rz(-0.19721017091099396) q[5];
ry(-3.1350349286568178) q[6];
rz(1.3161317279169198) q[6];
ry(-3.1312203086038695) q[7];
rz(0.6162977083083185) q[7];
ry(-1.595277330040121) q[8];
rz(-0.2407793242625374) q[8];
ry(-1.55378254795929) q[9];
rz(0.0210776342022068) q[9];
ry(1.570341923597618) q[10];
rz(0.12897081309631364) q[10];
ry(1.5676268980132058) q[11];
rz(0.04831910204613392) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.13416043696233243) q[0];
rz(3.0094028820002507) q[0];
ry(-1.6268403242270404) q[1];
rz(-1.7586077586914381) q[1];
ry(1.6460075454019927) q[2];
rz(1.6852814216794052) q[2];
ry(3.133962733112445) q[3];
rz(-2.2694290880785166) q[3];
ry(-0.5548063042322138) q[4];
rz(-1.5446409001027677) q[4];
ry(-1.9196894042566903) q[5];
rz(-0.7360015343920527) q[5];
ry(-2.3979301719473103) q[6];
rz(0.9780589164826459) q[6];
ry(2.113024943097634) q[7];
rz(-2.4280838938024787) q[7];
ry(1.5858282991773285) q[8];
rz(1.637741524300136) q[8];
ry(1.3152727617648896) q[9];
rz(-1.0650617161571851) q[9];
ry(0.001824238295037972) q[10];
rz(-0.12580148265566926) q[10];
ry(-0.018582940576216556) q[11];
rz(0.06554788910681013) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.134082263245971) q[0];
rz(-1.1005187000559187) q[0];
ry(1.583757788966701) q[1];
rz(1.591295754933772) q[1];
ry(-3.131752832418008) q[2];
rz(1.6802410214887684) q[2];
ry(-3.1388011528600184) q[3];
rz(-0.5304317195080058) q[3];
ry(0.0007970691141217931) q[4];
rz(-1.5595500803904834) q[4];
ry(0.0023849197267272974) q[5];
rz(0.18504999650354395) q[5];
ry(0.028828227716978728) q[6];
rz(-1.7808714772056542) q[6];
ry(-1.5998350293294397) q[7];
rz(-0.7211061788620289) q[7];
ry(0.2999907748392907) q[8];
rz(1.5426138073311244) q[8];
ry(0.0018937552136222052) q[9];
rz(-2.0978248507085993) q[9];
ry(-0.9719432078864669) q[10];
rz(1.5842886638715492) q[10];
ry(2.967084827525798) q[11];
rz(-1.4606373580220033) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5387925570656642) q[0];
rz(1.5851677673174298) q[0];
ry(1.5939753302549773) q[1];
rz(-0.9833709477150685) q[1];
ry(-1.409480510826499) q[2];
rz(2.7355412879162886) q[2];
ry(0.2231101416228703) q[3];
rz(0.33749167020516746) q[3];
ry(-3.118309352567855) q[4];
rz(-1.382478922512581) q[4];
ry(0.481735063464132) q[5];
rz(-0.15368660770170947) q[5];
ry(-3.1332606273271044) q[6];
rz(0.4753920607334725) q[6];
ry(-1.5963977493310064) q[7];
rz(-0.4818285203697185) q[7];
ry(3.1410059596236564) q[8];
rz(1.7835881845394348) q[8];
ry(3.1410184112464767) q[9];
rz(1.5994298848736515) q[9];
ry(-0.3991601725529697) q[10];
rz(-0.01772884515471206) q[10];
ry(-1.9489586041099987) q[11];
rz(-3.1227903239821444) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.6553028125703406) q[0];
rz(-1.753604520766721) q[0];
ry(-3.1261063487923972) q[1];
rz(0.7959483860567795) q[1];
ry(3.1347219454618336) q[2];
rz(-0.40981034180349) q[2];
ry(3.118424953166223) q[3];
rz(-2.4527339826600514) q[3];
ry(-1.569668850709222) q[4];
rz(-3.1296621820141537) q[4];
ry(1.5724303985037134) q[5];
rz(3.113861195063495) q[5];
ry(-3.0959026036664206) q[6];
rz(-1.8668685382216381) q[6];
ry(-3.0303559327285843) q[7];
rz(-0.5077160502916129) q[7];
ry(-0.009015881770290215) q[8];
rz(2.9326283165152507) q[8];
ry(1.5669896387087483) q[9];
rz(1.57178548764381) q[9];
ry(-1.5573900105862304) q[10];
rz(0.21987656145501178) q[10];
ry(-1.5701281166909435) q[11];
rz(-0.7084049646944584) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5480090236237756) q[0];
rz(-0.3839970882690489) q[0];
ry(1.722411886200038) q[1];
rz(-2.8795149532188087) q[1];
ry(1.517809449482144) q[2];
rz(-1.3993919565664736) q[2];
ry(0.011754760917562555) q[3];
rz(3.05464636365869) q[3];
ry(1.5654187594228255) q[4];
rz(2.556382503573931) q[4];
ry(1.6004934850563997) q[5];
rz(-0.32449765820579696) q[5];
ry(1.5862795915218177) q[6];
rz(-1.571546760116617) q[6];
ry(1.5765357272667193) q[7];
rz(2.3900333744074937) q[7];
ry(-1.5705347278436674) q[8];
rz(-2.4462478249519437) q[8];
ry(-2.29953417081014) q[9];
rz(0.02100499009364576) q[9];
ry(3.0378581148960757) q[10];
rz(-2.581298247864157) q[10];
ry(0.195299766408092) q[11];
rz(0.7593674420003713) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.598057062382229) q[0];
rz(-0.5252496325940488) q[0];
ry(-1.5200491121835042) q[1];
rz(1.087003717762878) q[1];
ry(3.1005933559238956) q[2];
rz(-2.308491770890817) q[2];
ry(-0.0717965885427953) q[3];
rz(-1.8168078907882386) q[3];
ry(0.007215295242569475) q[4];
rz(0.5628980620851758) q[4];
ry(-3.121032987399434) q[5];
rz(-2.8482082485539566) q[5];
ry(-2.6316338104450723) q[6];
rz(-0.5081564749478718) q[6];
ry(-3.140461827535882) q[7];
rz(-2.312045745219907) q[7];
ry(-0.002369318978627355) q[8];
rz(1.7460235624103628) q[8];
ry(-0.14802203961449667) q[9];
rz(0.6585004038446938) q[9];
ry(0.0025049239119501266) q[10];
rz(0.6843535022941838) q[10];
ry(0.005213357000947383) q[11];
rz(1.6355290822968227) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.924355779806466) q[0];
rz(2.5986188439200975) q[0];
ry(2.9603387718187144) q[1];
rz(-0.6225324018038529) q[1];
ry(-3.126668996289475) q[2];
rz(1.8417966666338348) q[2];
ry(3.140213014994906) q[3];
rz(3.027802169412915) q[3];
ry(1.5609397637218874) q[4];
rz(-0.5815407983066274) q[4];
ry(-1.6103189994029616) q[5];
rz(-0.5202392139721308) q[5];
ry(-0.03435114889545642) q[6];
rz(0.014171184544537082) q[6];
ry(1.5433489304913048) q[7];
rz(-0.0015293219300951723) q[7];
ry(3.1269725243432176) q[8];
rz(-2.9073591767206044) q[8];
ry(-3.1120392994487918) q[9];
rz(-2.4374704800568754) q[9];
ry(3.069608230597713) q[10];
rz(0.2527173020350932) q[10];
ry(1.6735508156149113) q[11];
rz(2.3038154346816446) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.5238679367370582) q[0];
rz(2.1846670627511817) q[0];
ry(-0.38192105781659) q[1];
rz(-3.1222936265683554) q[1];
ry(3.1414339798965005) q[2];
rz(-2.6929990209104613) q[2];
ry(3.1373792913169947) q[3];
rz(0.02728114365721357) q[3];
ry(0.009682662112415849) q[4];
rz(0.5714015866326223) q[4];
ry(-0.003734990866036902) q[5];
rz(-2.6382610662258497) q[5];
ry(3.0834371624892167) q[6];
rz(-2.465990318600067) q[6];
ry(2.5549065421366146) q[7];
rz(-0.2113055480290214) q[7];
ry(3.1386493168116147) q[8];
rz(-2.203575173835846) q[8];
ry(-0.45716685333452484) q[9];
rz(3.11730202233127) q[9];
ry(-3.140400941644402) q[10];
rz(2.503378891048019) q[10];
ry(-0.0010780260944045336) q[11];
rz(2.4029652302095) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.3403344931508192) q[0];
rz(-0.821486598660357) q[0];
ry(0.1813094222384773) q[1];
rz(2.3930603152487673) q[1];
ry(-3.115049777864614) q[2];
rz(-0.8673427972012088) q[2];
ry(1.5552565670134) q[3];
rz(3.1409935231268267) q[3];
ry(1.5747204574646427) q[4];
rz(1.6237084354284923) q[4];
ry(-1.5679565471449706) q[5];
rz(-3.050205598086599) q[5];
ry(3.1359861577598487) q[6];
rz(-0.4025812589090281) q[6];
ry(-0.021389280822427637) q[7];
rz(1.7670023029881259) q[7];
ry(-1.5892211380203491) q[8];
rz(-1.5703186229786068) q[8];
ry(-1.5797066334690113) q[9];
rz(-1.5632287279026478) q[9];
ry(-1.6161200746960904) q[10];
rz(1.7338554999971088) q[10];
ry(-1.5686901566712153) q[11];
rz(-3.0996566773577294) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.0028763889233071726) q[0];
rz(-3.0741575979329814) q[0];
ry(3.1410611977884835) q[1];
rz(1.9478285357003788) q[1];
ry(1.5660440456597051) q[2];
rz(-1.5762782966397229) q[2];
ry(-1.5801605458516343) q[3];
rz(-1.8815555483552373) q[3];
ry(-3.1411864298419645) q[4];
rz(0.05491908832930038) q[4];
ry(2.790763626626351) q[5];
rz(0.47960964895548375) q[5];
ry(-1.5696798129795653) q[6];
rz(1.2255805203433328) q[6];
ry(-1.5689804804055993) q[7];
rz(3.134581508404283) q[7];
ry(1.5724113018837018) q[8];
rz(-0.0017706622724888987) q[8];
ry(-1.5703556033125743) q[9];
rz(-1.5721112211236168) q[9];
ry(1.5706458467876852) q[10];
rz(-3.1343737927783777) q[10];
ry(-1.5761689672401724) q[11];
rz(-1.5661760692640367) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(3.139383917126865) q[0];
rz(2.33918377824079) q[0];
ry(3.1409378597838464) q[1];
rz(1.4213123105299796) q[1];
ry(-1.5767648454627885) q[2];
rz(-1.5813615407245911) q[2];
ry(3.1319870353711847) q[3];
rz(0.5013773327203791) q[3];
ry(-0.0005841167866545798) q[4];
rz(1.9708239488769888) q[4];
ry(-3.1405808805382014) q[5];
rz(-2.8056310573008916) q[5];
ry(3.0663901600636954) q[6];
rz(-0.8819162128636998) q[6];
ry(-1.5824316020547977) q[7];
rz(0.4600768161762839) q[7];
ry(-1.5707065490085133) q[8];
rz(-1.5725112352014756) q[8];
ry(-1.1149987824580914) q[9];
rz(1.0372957199024258) q[9];
ry(1.5678956636617474) q[10];
rz(2.962253642812643) q[10];
ry(1.577249064925021) q[11];
rz(1.5735330803666434) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5718682652869294) q[0];
rz(0.01933966428647871) q[0];
ry(0.018614950193077462) q[1];
rz(1.5496759213804525) q[1];
ry(-1.5844276738192669) q[2];
rz(0.8702635393447063) q[2];
ry(3.130461890619221) q[3];
rz(-2.473129873108724) q[3];
ry(-3.141048836201953) q[4];
rz(-0.0718643456006367) q[4];
ry(-0.0002947408576157901) q[5];
rz(-2.97755183728245) q[5];
ry(-3.1403768580781457) q[6];
rz(2.6075525641412907) q[6];
ry(3.1415191746943076) q[7];
rz(-2.5717941443056893) q[7];
ry(1.5537908765859156) q[8];
rz(-1.5748812324315153) q[8];
ry(-0.0009024541341208131) q[9];
rz(2.3167969170479266) q[9];
ry(1.575421165351946) q[10];
rz(1.760293673566621) q[10];
ry(1.5688886444939891) q[11];
rz(-1.8361922530256063) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5711147199323097) q[0];
rz(-1.3181854985806158) q[0];
ry(0.027984617456957572) q[1];
rz(1.1438558116219903) q[1];
ry(0.01564550824197491) q[2];
rz(-2.3272846781044114) q[2];
ry(-1.5740510027500791) q[3];
rz(-1.8378093442003811) q[3];
ry(0.006625732348055635) q[4];
rz(-2.365774602963674) q[4];
ry(1.5659869241244584) q[5];
rz(-0.26697375599983303) q[5];
ry(1.5758830845242666) q[6];
rz(1.214756127979424) q[6];
ry(0.0012416810875972573) q[7];
rz(-1.9165470458487317) q[7];
ry(1.5685795565227205) q[8];
rz(1.2067920554651075) q[8];
ry(-3.1390316834823566) q[9];
rz(1.5540807522408882) q[9];
ry(-0.0038661539010957924) q[10];
rz(1.0158263285766598) q[10];
ry(0.00577310359072758) q[11];
rz(-3.1004167979136676) q[11];