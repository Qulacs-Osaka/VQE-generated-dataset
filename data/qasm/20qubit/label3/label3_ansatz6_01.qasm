OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.9966367378796415) q[0];
ry(2.1298878746422085) q[1];
cx q[0],q[1];
ry(1.4616984264293618) q[0];
ry(1.6088785347586783) q[1];
cx q[0],q[1];
ry(1.0561184900676182) q[1];
ry(1.3347002572181308) q[2];
cx q[1],q[2];
ry(1.410707533153503) q[1];
ry(-1.8299921778767863) q[2];
cx q[1],q[2];
ry(-1.5035281537191132) q[2];
ry(-2.4962504312974345) q[3];
cx q[2],q[3];
ry(1.1537525854528106) q[2];
ry(-2.0749639034556617) q[3];
cx q[2],q[3];
ry(-1.7361325397919867) q[3];
ry(-2.774194417680475) q[4];
cx q[3],q[4];
ry(-1.2052408209452716) q[3];
ry(-2.562484784525193) q[4];
cx q[3],q[4];
ry(2.574542792461647) q[4];
ry(-0.08551625491004308) q[5];
cx q[4],q[5];
ry(-1.4723332469248256) q[4];
ry(-1.8453658693326795) q[5];
cx q[4],q[5];
ry(-1.1071351975468353) q[5];
ry(2.8980586509060053) q[6];
cx q[5],q[6];
ry(-1.8007070585210085) q[5];
ry(-1.3192357735727347) q[6];
cx q[5],q[6];
ry(-3.0830720855897225) q[6];
ry(3.138059584656324) q[7];
cx q[6],q[7];
ry(1.5829334428302573) q[6];
ry(1.4730850124795207) q[7];
cx q[6],q[7];
ry(0.371820843110191) q[7];
ry(1.8929730437540346) q[8];
cx q[7],q[8];
ry(-1.2302965935712933) q[7];
ry(2.6399670343098154) q[8];
cx q[7],q[8];
ry(1.1360535898920154) q[8];
ry(2.6429029144142846) q[9];
cx q[8],q[9];
ry(1.790285722663021) q[8];
ry(1.595723567786961) q[9];
cx q[8],q[9];
ry(0.2485833122531368) q[9];
ry(-0.6578441638431334) q[10];
cx q[9],q[10];
ry(-0.1931937327052798) q[9];
ry(2.6705223367129225) q[10];
cx q[9],q[10];
ry(-2.7464079309625147) q[10];
ry(-2.385111086779061) q[11];
cx q[10],q[11];
ry(-1.9268384450181228) q[10];
ry(-2.3851404319834337) q[11];
cx q[10],q[11];
ry(-1.5448116304150485) q[11];
ry(-0.5232793635611006) q[12];
cx q[11],q[12];
ry(1.7777016495076896) q[11];
ry(2.130180153530045) q[12];
cx q[11],q[12];
ry(0.03864524373103342) q[12];
ry(-3.1391443929979617) q[13];
cx q[12],q[13];
ry(1.5163735308584378) q[12];
ry(-1.5303510493185648) q[13];
cx q[12],q[13];
ry(-1.2094147008037786) q[13];
ry(-0.6512847908553699) q[14];
cx q[13],q[14];
ry(-2.5034208609476183) q[13];
ry(-1.9693857303237046) q[14];
cx q[13],q[14];
ry(-2.599760193505422) q[14];
ry(2.917935142965831) q[15];
cx q[14],q[15];
ry(-1.756059305456433) q[14];
ry(1.3496380438870967) q[15];
cx q[14],q[15];
ry(-3.002479988340353) q[15];
ry(0.17425176857653932) q[16];
cx q[15],q[16];
ry(0.37194444394615633) q[15];
ry(-0.848254575497335) q[16];
cx q[15],q[16];
ry(2.890113133270442) q[16];
ry(-0.22018122861725864) q[17];
cx q[16],q[17];
ry(-0.44664697478058635) q[16];
ry(-2.7161797600592203) q[17];
cx q[16],q[17];
ry(-0.821644720343607) q[17];
ry(-2.9760957358740763) q[18];
cx q[17],q[18];
ry(-1.4073808023290555) q[17];
ry(1.7067367991813116) q[18];
cx q[17],q[18];
ry(-0.9674029910539907) q[18];
ry(-0.29593370522904183) q[19];
cx q[18],q[19];
ry(-1.1313421246806261) q[18];
ry(-0.0558637049774802) q[19];
cx q[18],q[19];
ry(2.1093351062169092) q[0];
ry(-1.5765245252462148) q[1];
cx q[0],q[1];
ry(1.6248383574508996) q[0];
ry(-1.7617083242292875) q[1];
cx q[0],q[1];
ry(-2.402320881407634) q[1];
ry(0.15891035467002368) q[2];
cx q[1],q[2];
ry(0.20423252475046816) q[1];
ry(1.2532659559764863) q[2];
cx q[1],q[2];
ry(2.3958450104221964) q[2];
ry(2.7192875375987193) q[3];
cx q[2],q[3];
ry(2.937432967538144) q[2];
ry(-1.7502403639793052) q[3];
cx q[2],q[3];
ry(-1.3165452556663892) q[3];
ry(2.496975244530228) q[4];
cx q[3],q[4];
ry(2.1888193227037025) q[3];
ry(1.5875755331043517) q[4];
cx q[3],q[4];
ry(-0.18974071486517927) q[4];
ry(-1.4074391152223138) q[5];
cx q[4],q[5];
ry(-2.7733770712349495) q[4];
ry(-2.881002251933573) q[5];
cx q[4],q[5];
ry(0.21875595840989226) q[5];
ry(-1.6010109275696758) q[6];
cx q[5],q[6];
ry(-0.18306817971804185) q[5];
ry(-1.611667940590019) q[6];
cx q[5],q[6];
ry(0.46691935505526844) q[6];
ry(0.028062024453182843) q[7];
cx q[6],q[7];
ry(-1.518125746569285) q[6];
ry(1.551701029663712) q[7];
cx q[6],q[7];
ry(1.3506254241185118) q[7];
ry(0.0008892803717106545) q[8];
cx q[7],q[8];
ry(1.568614740913331) q[7];
ry(-3.1291715731291396) q[8];
cx q[7],q[8];
ry(-0.2131322836982264) q[8];
ry(2.8778382729670637) q[9];
cx q[8],q[9];
ry(1.8017370550202718) q[8];
ry(-1.4968930168359) q[9];
cx q[8],q[9];
ry(-2.7288039031157254) q[9];
ry(1.9360116905390106) q[10];
cx q[9],q[10];
ry(2.9649945955155057) q[9];
ry(-0.5803190213824614) q[10];
cx q[9],q[10];
ry(-3.048920796872912) q[10];
ry(-0.7870348515726207) q[11];
cx q[10],q[11];
ry(0.6118151546859525) q[10];
ry(2.9692919314942343) q[11];
cx q[10],q[11];
ry(-0.1273483875272321) q[11];
ry(3.124119584945485) q[12];
cx q[11],q[12];
ry(0.5708491529277219) q[11];
ry(-0.5429414807440271) q[12];
cx q[11],q[12];
ry(2.647021336842465) q[12];
ry(0.1304691078757454) q[13];
cx q[12],q[13];
ry(1.5417639314270577) q[12];
ry(-1.5303377978748092) q[13];
cx q[12],q[13];
ry(2.4900292992579067) q[13];
ry(-2.8805052399707005) q[14];
cx q[13],q[14];
ry(0.34344934291084234) q[13];
ry(-0.24152618617378538) q[14];
cx q[13],q[14];
ry(2.806433002320223) q[14];
ry(-1.3084869165237052) q[15];
cx q[14],q[15];
ry(1.4692494002683347) q[14];
ry(-2.9909577725318335) q[15];
cx q[14],q[15];
ry(-1.5912960498903344) q[15];
ry(-0.7346244550252286) q[16];
cx q[15],q[16];
ry(0.10068060994096317) q[15];
ry(-0.7045167159799769) q[16];
cx q[15],q[16];
ry(-0.2679077335996096) q[16];
ry(1.742296362159812) q[17];
cx q[16],q[17];
ry(-2.364286726097019) q[16];
ry(-1.5693250735244926) q[17];
cx q[16],q[17];
ry(1.835015549739465) q[17];
ry(-2.6691597091139108) q[18];
cx q[17],q[18];
ry(-0.6651688715069884) q[17];
ry(2.645023835222686) q[18];
cx q[17],q[18];
ry(1.7788852129086772) q[18];
ry(-2.29739180412128) q[19];
cx q[18],q[19];
ry(2.5109083804155197) q[18];
ry(2.5902640878963967) q[19];
cx q[18],q[19];
ry(0.04925170193368) q[0];
ry(-2.3603841271372397) q[1];
cx q[0],q[1];
ry(2.4193142940659307) q[0];
ry(-0.9636993070287998) q[1];
cx q[0],q[1];
ry(2.5785209288071296) q[1];
ry(-0.8730867077274205) q[2];
cx q[1],q[2];
ry(-1.331431304896957) q[1];
ry(1.8794342002531823) q[2];
cx q[1],q[2];
ry(-0.7438194563692813) q[2];
ry(-2.5343117258475907) q[3];
cx q[2],q[3];
ry(-2.8455803230727694) q[2];
ry(1.5264402597733466) q[3];
cx q[2],q[3];
ry(-2.64443838090959) q[3];
ry(1.7654192591701316) q[4];
cx q[3],q[4];
ry(-2.747282718495059) q[3];
ry(2.9701241350370347) q[4];
cx q[3],q[4];
ry(0.1641602411166616) q[4];
ry(1.3480240403906236) q[5];
cx q[4],q[5];
ry(1.7405162878339449) q[4];
ry(0.1267108041078236) q[5];
cx q[4],q[5];
ry(1.2163520090852336) q[5];
ry(-0.007198030614034195) q[6];
cx q[5],q[6];
ry(-2.655470023645105) q[5];
ry(-0.04743159758377872) q[6];
cx q[5],q[6];
ry(0.011523166126429212) q[6];
ry(-1.4724082302774493) q[7];
cx q[6],q[7];
ry(1.5680344646276998) q[6];
ry(-1.5669312008211946) q[7];
cx q[6],q[7];
ry(0.04928753643739245) q[7];
ry(-1.751611363225389) q[8];
cx q[7],q[8];
ry(3.105780755537565) q[7];
ry(0.2539228705308373) q[8];
cx q[7],q[8];
ry(0.21489635183354763) q[8];
ry(1.395038705080245) q[9];
cx q[8],q[9];
ry(2.8959596452234178) q[8];
ry(2.8734713438617128) q[9];
cx q[8],q[9];
ry(-1.2019171667262682) q[9];
ry(-1.597722583895111) q[10];
cx q[9],q[10];
ry(1.5563391425744264) q[9];
ry(-1.0832641197398605) q[10];
cx q[9],q[10];
ry(1.645067504028475) q[10];
ry(-1.5477661434315415) q[11];
cx q[10],q[11];
ry(-1.6181392187740986) q[10];
ry(2.7832875886079194) q[11];
cx q[10],q[11];
ry(1.567486356939952) q[11];
ry(1.5927808519686866) q[12];
cx q[11],q[12];
ry(-1.5720840222030594) q[11];
ry(1.5882374792869438) q[12];
cx q[11],q[12];
ry(-1.4020730303250302) q[12];
ry(2.5752985749837047) q[13];
cx q[12],q[13];
ry(1.3098931584180245) q[12];
ry(-2.485416195689604) q[13];
cx q[12],q[13];
ry(3.086905540200299) q[13];
ry(3.0458071272049123) q[14];
cx q[13],q[14];
ry(1.5030692895103663) q[13];
ry(1.5245878795249688) q[14];
cx q[13],q[14];
ry(-3.1394888383639934) q[14];
ry(-1.2042742586804502) q[15];
cx q[14],q[15];
ry(1.5075874955013038) q[14];
ry(-0.5063985862430895) q[15];
cx q[14],q[15];
ry(-0.6971861126823002) q[15];
ry(0.11851375431338518) q[16];
cx q[15],q[16];
ry(1.9196585741554502) q[15];
ry(3.134333018598057) q[16];
cx q[15],q[16];
ry(-2.779399862629243) q[16];
ry(1.6254286894997616) q[17];
cx q[16],q[17];
ry(1.9468752398732423) q[16];
ry(0.03250674475826596) q[17];
cx q[16],q[17];
ry(-2.2951282422794685) q[17];
ry(-1.3874476761239087) q[18];
cx q[17],q[18];
ry(1.660102658573359) q[17];
ry(-3.0845824390281713) q[18];
cx q[17],q[18];
ry(1.7167367978934562) q[18];
ry(-1.7327465883532518) q[19];
cx q[18],q[19];
ry(1.5619809469111907) q[18];
ry(-2.0578414631543303) q[19];
cx q[18],q[19];
ry(-1.398686823418054) q[0];
ry(-2.9971087297061207) q[1];
cx q[0],q[1];
ry(1.8288310405287387) q[0];
ry(2.120843372921141) q[1];
cx q[0],q[1];
ry(0.2113886740694433) q[1];
ry(1.3917156713202476) q[2];
cx q[1],q[2];
ry(-0.46549423185713806) q[1];
ry(-1.5383634997592415) q[2];
cx q[1],q[2];
ry(-2.962507631517322) q[2];
ry(1.1362064702846482) q[3];
cx q[2],q[3];
ry(0.5654287451427856) q[2];
ry(1.5762748869916345) q[3];
cx q[2],q[3];
ry(-2.769187025631956) q[3];
ry(1.5354189628700412) q[4];
cx q[3],q[4];
ry(-1.6044961097325767) q[3];
ry(-1.5086885618902) q[4];
cx q[3],q[4];
ry(-0.16180522632066374) q[4];
ry(-0.795734718851101) q[5];
cx q[4],q[5];
ry(-3.0825625664749197) q[4];
ry(-0.09440366583808137) q[5];
cx q[4],q[5];
ry(-1.1392918870450837) q[5];
ry(2.0794304545883273) q[6];
cx q[5],q[6];
ry(-0.47107994219569616) q[5];
ry(-1.5659712643972405) q[6];
cx q[5],q[6];
ry(-3.1312452141698732) q[6];
ry(3.0822553185202763) q[7];
cx q[6],q[7];
ry(-1.5686139252979534) q[6];
ry(-1.5636854254885) q[7];
cx q[6],q[7];
ry(0.011421860297821006) q[7];
ry(1.8633504837018409) q[8];
cx q[7],q[8];
ry(3.1033717022638077) q[7];
ry(0.03457228265428974) q[8];
cx q[7],q[8];
ry(-1.0460780448650955) q[8];
ry(-0.04488736405383431) q[9];
cx q[8],q[9];
ry(-2.5973784906659763) q[8];
ry(0.14145149190493353) q[9];
cx q[8],q[9];
ry(0.19865800795783883) q[9];
ry(3.1338180623695453) q[10];
cx q[9],q[10];
ry(-1.597167896176353) q[9];
ry(1.5806907796932244) q[10];
cx q[9],q[10];
ry(2.5283339344505653) q[10];
ry(0.024931114523535667) q[11];
cx q[10],q[11];
ry(-2.4790777780147266) q[10];
ry(3.135482690868696) q[11];
cx q[10],q[11];
ry(0.04433321702752835) q[11];
ry(-0.08714829333026262) q[12];
cx q[11],q[12];
ry(-1.5606338110598825) q[11];
ry(1.57702646679767) q[12];
cx q[11],q[12];
ry(-3.1352875076853373) q[12];
ry(3.0446461657207116) q[13];
cx q[12],q[13];
ry(-0.4692749989899987) q[12];
ry(0.0792509319706447) q[13];
cx q[12],q[13];
ry(0.03799358682522073) q[13];
ry(-0.043412633917955155) q[14];
cx q[13],q[14];
ry(-1.598128610154913) q[13];
ry(1.5329798297029997) q[14];
cx q[13],q[14];
ry(-0.05441662434152378) q[14];
ry(-1.200818415991983) q[15];
cx q[14],q[15];
ry(0.12394083498010833) q[14];
ry(-1.6291819303514963) q[15];
cx q[14],q[15];
ry(0.04488259834025854) q[15];
ry(-0.3988701317205883) q[16];
cx q[15],q[16];
ry(-2.28589014534603) q[15];
ry(-2.861060236031375) q[16];
cx q[15],q[16];
ry(3.0742019911740717) q[16];
ry(0.822137391432992) q[17];
cx q[16],q[17];
ry(-2.3409969356041356) q[16];
ry(1.6237236411636617) q[17];
cx q[16],q[17];
ry(-0.05501538319557674) q[17];
ry(1.7446308224835059) q[18];
cx q[17],q[18];
ry(-1.3724604863169967) q[17];
ry(1.444676878106729) q[18];
cx q[17],q[18];
ry(-3.1392235166880225) q[18];
ry(-1.6548078377355602) q[19];
cx q[18],q[19];
ry(-2.6682892417920723) q[18];
ry(-1.6750736806402349) q[19];
cx q[18],q[19];
ry(-1.223150024823659) q[0];
ry(0.5656824403784446) q[1];
ry(-1.5066171832404258) q[2];
ry(-2.5568632500988686) q[3];
ry(2.0830374098331177) q[4];
ry(0.7676639597041653) q[5];
ry(0.30969039876399496) q[6];
ry(1.909311777829613) q[7];
ry(-1.8389973163057531) q[8];
ry(0.4773489315981694) q[9];
ry(-1.9363791824234524) q[10];
ry(0.4293082881066578) q[11];
ry(-1.076231771088146) q[12];
ry(0.4169264295550068) q[13];
ry(1.2980192343472798) q[14];
ry(-1.0948122928683564) q[15];
ry(-1.0030166848360231) q[16];
ry(2.0456055930832333) q[17];
ry(-1.005601437579523) q[18];
ry(-1.366126367006692) q[19];