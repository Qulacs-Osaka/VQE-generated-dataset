OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.923619817978509) q[0];
ry(-0.41751671219498643) q[1];
cx q[0],q[1];
ry(1.568522781387606) q[0];
ry(-1.6808260539980795) q[1];
cx q[0],q[1];
ry(1.7745889770587882) q[2];
ry(2.407565959964829) q[3];
cx q[2],q[3];
ry(2.2332927791705033) q[2];
ry(-1.5402945321612966) q[3];
cx q[2],q[3];
ry(1.249603585412535) q[4];
ry(1.4601831164736) q[5];
cx q[4],q[5];
ry(-1.894294761995492) q[4];
ry(-1.8701740788973142) q[5];
cx q[4],q[5];
ry(0.3624799918936652) q[6];
ry(-2.889040958793238) q[7];
cx q[6],q[7];
ry(2.085292168541046) q[6];
ry(0.9394583732744017) q[7];
cx q[6],q[7];
ry(-2.1824468523757825) q[8];
ry(2.7743048479261816) q[9];
cx q[8],q[9];
ry(0.18524612457353307) q[8];
ry(0.16509569802313975) q[9];
cx q[8],q[9];
ry(-2.808131091216832) q[10];
ry(1.4292874207523791) q[11];
cx q[10],q[11];
ry(2.909528980264829) q[10];
ry(-2.7385709679290664) q[11];
cx q[10],q[11];
ry(1.6549873176901728) q[0];
ry(-1.3376832966682528) q[2];
cx q[0],q[2];
ry(-2.0543130022030103) q[0];
ry(0.3537600538979202) q[2];
cx q[0],q[2];
ry(-0.7322300575843839) q[2];
ry(-1.3306898330618928) q[4];
cx q[2],q[4];
ry(2.1240695665222677) q[2];
ry(-1.9022371423007869) q[4];
cx q[2],q[4];
ry(0.00817756120231472) q[4];
ry(-0.1440239704197809) q[6];
cx q[4],q[6];
ry(1.53941678857261) q[4];
ry(0.501854314253297) q[6];
cx q[4],q[6];
ry(1.5261430128677045) q[6];
ry(0.4295097100818239) q[8];
cx q[6],q[8];
ry(-1.7477056235230735) q[6];
ry(-2.862467630237185) q[8];
cx q[6],q[8];
ry(2.0642321183260517) q[8];
ry(-2.624144869626641) q[10];
cx q[8],q[10];
ry(2.9576497765804044) q[8];
ry(-1.0261082764539688) q[10];
cx q[8],q[10];
ry(0.40698173183533604) q[1];
ry(2.86924045222818) q[3];
cx q[1],q[3];
ry(1.4275053934784028) q[1];
ry(-0.963186663714712) q[3];
cx q[1],q[3];
ry(-1.3231415594792997) q[3];
ry(2.7359105043564957) q[5];
cx q[3],q[5];
ry(0.26148035429398675) q[3];
ry(2.7153064869780548) q[5];
cx q[3],q[5];
ry(-2.6344905880766754) q[5];
ry(2.6456416717996096) q[7];
cx q[5],q[7];
ry(-3.1381735940238653) q[5];
ry(0.020786567237103704) q[7];
cx q[5],q[7];
ry(-0.43646549131568424) q[7];
ry(2.013238727356029) q[9];
cx q[7],q[9];
ry(-2.2728326579600715) q[7];
ry(-0.025676785549457203) q[9];
cx q[7],q[9];
ry(1.6826451540858076) q[9];
ry(2.778039360338174) q[11];
cx q[9],q[11];
ry(3.03475715844233) q[9];
ry(0.9863505982414164) q[11];
cx q[9],q[11];
ry(-2.6801587778816294) q[0];
ry(-0.22894899602775) q[3];
cx q[0],q[3];
ry(-1.3782534273164204) q[0];
ry(-2.420172069016167) q[3];
cx q[0],q[3];
ry(-2.2161220924644582) q[1];
ry(-1.891081561812972) q[2];
cx q[1],q[2];
ry(-2.300496706279254) q[1];
ry(1.0522917444250945) q[2];
cx q[1],q[2];
ry(-3.0883109181232404) q[2];
ry(-1.2560083929099677) q[5];
cx q[2],q[5];
ry(-1.2441566749801813) q[2];
ry(-0.7223241217151992) q[5];
cx q[2],q[5];
ry(-0.7438895172550523) q[3];
ry(-0.9785461582007763) q[4];
cx q[3],q[4];
ry(-1.1960185653472812) q[3];
ry(-2.4295497106552264) q[4];
cx q[3],q[4];
ry(-2.4918276834555724) q[4];
ry(1.0466499225633872) q[7];
cx q[4],q[7];
ry(-2.6130268283638465) q[4];
ry(-2.9753645499447168) q[7];
cx q[4],q[7];
ry(-2.8660847088819588) q[5];
ry(0.6469910176460116) q[6];
cx q[5],q[6];
ry(-0.40688369410509473) q[5];
ry(1.9471725847580188) q[6];
cx q[5],q[6];
ry(-2.826386538548427) q[6];
ry(1.5095837224523014) q[9];
cx q[6],q[9];
ry(-2.2041856355105405) q[6];
ry(0.029667856722698893) q[9];
cx q[6],q[9];
ry(0.40451847297528154) q[7];
ry(-1.6840603081396708) q[8];
cx q[7],q[8];
ry(-0.631517437999964) q[7];
ry(-3.1239821469005387) q[8];
cx q[7],q[8];
ry(1.2644044344103744) q[8];
ry(-0.1775002059428912) q[11];
cx q[8],q[11];
ry(1.323778243909198) q[8];
ry(1.7341768655097074) q[11];
cx q[8],q[11];
ry(-2.0591541605723185) q[9];
ry(-2.0222138203925217) q[10];
cx q[9],q[10];
ry(-2.000463391797939) q[9];
ry(1.4569068123734261) q[10];
cx q[9],q[10];
ry(-1.2939215102022734) q[0];
ry(1.19214379651575) q[1];
cx q[0],q[1];
ry(2.4159326879130005) q[0];
ry(-1.0126693862643785) q[1];
cx q[0],q[1];
ry(-2.581499907333408) q[2];
ry(2.542593887926947) q[3];
cx q[2],q[3];
ry(-2.5733272225035773) q[2];
ry(0.6866358256216207) q[3];
cx q[2],q[3];
ry(2.486166564208065) q[4];
ry(0.2312464396705538) q[5];
cx q[4],q[5];
ry(-0.9971788546674258) q[4];
ry(-2.117589699362599) q[5];
cx q[4],q[5];
ry(-2.30508326983585) q[6];
ry(-2.0531219699301015) q[7];
cx q[6],q[7];
ry(-2.824155609281709) q[6];
ry(-1.5441909701509313) q[7];
cx q[6],q[7];
ry(-2.3424506370012077) q[8];
ry(-2.723644011606348) q[9];
cx q[8],q[9];
ry(2.604684086414759) q[8];
ry(-2.7503194199002206) q[9];
cx q[8],q[9];
ry(-2.027285536005118) q[10];
ry(2.397013149607902) q[11];
cx q[10],q[11];
ry(1.8182081764902955) q[10];
ry(-2.3959216492470556) q[11];
cx q[10],q[11];
ry(0.008932281762949579) q[0];
ry(0.06476894639654951) q[2];
cx q[0],q[2];
ry(1.8516019648621829) q[0];
ry(-1.3487454021352356) q[2];
cx q[0],q[2];
ry(-0.334585959092081) q[2];
ry(-2.424090424728489) q[4];
cx q[2],q[4];
ry(1.7153533808205155) q[2];
ry(1.3274288758262167) q[4];
cx q[2],q[4];
ry(1.542086976895822) q[4];
ry(-1.3913662081547384) q[6];
cx q[4],q[6];
ry(-0.0924599466868799) q[4];
ry(-2.783907237154601) q[6];
cx q[4],q[6];
ry(-2.6370717376758748) q[6];
ry(-2.9343584965587457) q[8];
cx q[6],q[8];
ry(0.10411703428364769) q[6];
ry(-3.121130971250866) q[8];
cx q[6],q[8];
ry(2.053740254412876) q[8];
ry(-1.1845969112841799) q[10];
cx q[8],q[10];
ry(1.2816719528414993) q[8];
ry(1.1028332326525803) q[10];
cx q[8],q[10];
ry(2.7473345673179366) q[1];
ry(-1.69402185003016) q[3];
cx q[1],q[3];
ry(-2.657289880274302) q[1];
ry(1.7559143927943985) q[3];
cx q[1],q[3];
ry(2.7255363540227826) q[3];
ry(2.34788521858242) q[5];
cx q[3],q[5];
ry(1.8845568832714177) q[3];
ry(1.5324639369933777) q[5];
cx q[3],q[5];
ry(1.0783092706547492) q[5];
ry(2.94859784738727) q[7];
cx q[5],q[7];
ry(0.16102010189681512) q[5];
ry(-2.847356647475261) q[7];
cx q[5],q[7];
ry(-3.116710001324291) q[7];
ry(-2.7185301800193624) q[9];
cx q[7],q[9];
ry(2.552215953275982) q[7];
ry(0.010172899919541326) q[9];
cx q[7],q[9];
ry(0.659755588314483) q[9];
ry(-0.28202255289841094) q[11];
cx q[9],q[11];
ry(-2.432467987035547) q[9];
ry(0.2953518797360095) q[11];
cx q[9],q[11];
ry(0.15530764842351824) q[0];
ry(2.0044709690586715) q[3];
cx q[0],q[3];
ry(1.3387579770460567) q[0];
ry(1.6910602960115242) q[3];
cx q[0],q[3];
ry(-2.6345427050302828) q[1];
ry(0.6246059442892546) q[2];
cx q[1],q[2];
ry(-2.684989405210713) q[1];
ry(1.013725310120158) q[2];
cx q[1],q[2];
ry(-3.0912841557362265) q[2];
ry(-1.5086913070336259) q[5];
cx q[2],q[5];
ry(2.3324111218995154) q[2];
ry(-1.3073475123821512) q[5];
cx q[2],q[5];
ry(2.1540296850596623) q[3];
ry(-1.3558288063059585) q[4];
cx q[3],q[4];
ry(-1.1143477334929637) q[3];
ry(-1.61168697416637) q[4];
cx q[3],q[4];
ry(-1.2779849302133475) q[4];
ry(-2.8735732071212676) q[7];
cx q[4],q[7];
ry(0.2146799381438614) q[4];
ry(0.8805047089127971) q[7];
cx q[4],q[7];
ry(0.002091084124495879) q[5];
ry(0.4647819093778265) q[6];
cx q[5],q[6];
ry(-0.00026498635707739027) q[5];
ry(-2.8244172401718433) q[6];
cx q[5],q[6];
ry(2.4507171777078973) q[6];
ry(-0.01750287092044704) q[9];
cx q[6],q[9];
ry(-1.620045110218629) q[6];
ry(0.009087137462896246) q[9];
cx q[6],q[9];
ry(0.7217704409554458) q[7];
ry(2.9068597061016357) q[8];
cx q[7],q[8];
ry(-2.690736425209927) q[7];
ry(-0.05691114895816146) q[8];
cx q[7],q[8];
ry(-0.18931904905879876) q[8];
ry(-3.0553750182546535) q[11];
cx q[8],q[11];
ry(-0.2515686899999432) q[8];
ry(1.2142307026371568) q[11];
cx q[8],q[11];
ry(0.655348452607087) q[9];
ry(1.7172500979900986) q[10];
cx q[9],q[10];
ry(-0.8239817502859071) q[9];
ry(0.9489948084607593) q[10];
cx q[9],q[10];
ry(-0.020633712293955116) q[0];
ry(1.0880893707398371) q[1];
cx q[0],q[1];
ry(2.3145369816836205) q[0];
ry(0.41236156615045694) q[1];
cx q[0],q[1];
ry(-1.5927544880854376) q[2];
ry(-1.8162447953177434) q[3];
cx q[2],q[3];
ry(-1.067850186228097) q[2];
ry(0.5026220435393718) q[3];
cx q[2],q[3];
ry(-0.7309509693579934) q[4];
ry(-2.732555418014058) q[5];
cx q[4],q[5];
ry(2.395813910063253) q[4];
ry(-2.813270678318952) q[5];
cx q[4],q[5];
ry(0.9275838070511225) q[6];
ry(-2.234143297740654) q[7];
cx q[6],q[7];
ry(-0.1598484175934285) q[6];
ry(1.5180323114516252) q[7];
cx q[6],q[7];
ry(-2.456780728526009) q[8];
ry(-2.2343302290182256) q[9];
cx q[8],q[9];
ry(1.2325702063539943) q[8];
ry(3.133794476670487) q[9];
cx q[8],q[9];
ry(-2.803971098674773) q[10];
ry(-2.8548443588210883) q[11];
cx q[10],q[11];
ry(2.587175419240795) q[10];
ry(-1.9095685361627082) q[11];
cx q[10],q[11];
ry(2.2963049979778876) q[0];
ry(-0.09395448259697403) q[2];
cx q[0],q[2];
ry(-0.8763389129119444) q[0];
ry(-2.479708634966328) q[2];
cx q[0],q[2];
ry(3.1325031476497838) q[2];
ry(0.6057106973546071) q[4];
cx q[2],q[4];
ry(0.0704174085393575) q[2];
ry(-2.55358846667963) q[4];
cx q[2],q[4];
ry(0.18479988104226472) q[4];
ry(-2.796440734572883) q[6];
cx q[4],q[6];
ry(-3.020773876681283) q[4];
ry(0.010798267687358762) q[6];
cx q[4],q[6];
ry(-1.3433454504103626) q[6];
ry(3.069968109630423) q[8];
cx q[6],q[8];
ry(3.038141639432784) q[6];
ry(-2.92915329013341) q[8];
cx q[6],q[8];
ry(1.9902638861166733) q[8];
ry(2.918768345871152) q[10];
cx q[8],q[10];
ry(-2.292957710394275) q[8];
ry(1.917697253715344) q[10];
cx q[8],q[10];
ry(1.826806747162371) q[1];
ry(1.6073296009615634) q[3];
cx q[1],q[3];
ry(1.7796373927119244) q[1];
ry(-1.895783367607093) q[3];
cx q[1],q[3];
ry(-1.6742906027067486) q[3];
ry(3.115028708082504) q[5];
cx q[3],q[5];
ry(0.9065018521306882) q[3];
ry(-1.381924417901124) q[5];
cx q[3],q[5];
ry(0.3298160512055073) q[5];
ry(-1.3077406131527354) q[7];
cx q[5],q[7];
ry(0.028473134286005397) q[5];
ry(0.12119118144610468) q[7];
cx q[5],q[7];
ry(-0.07744082094823668) q[7];
ry(1.9232503885798753) q[9];
cx q[7],q[9];
ry(-2.176390856398253) q[7];
ry(-3.0721770086034126) q[9];
cx q[7],q[9];
ry(2.098244295840562) q[9];
ry(-1.519931444520911) q[11];
cx q[9],q[11];
ry(2.5368363107333445) q[9];
ry(-2.6328847734378407) q[11];
cx q[9],q[11];
ry(2.4270037286771395) q[0];
ry(-0.5760759768239415) q[3];
cx q[0],q[3];
ry(-0.04774625331040916) q[0];
ry(-0.9821719324110861) q[3];
cx q[0],q[3];
ry(1.991562519823997) q[1];
ry(-1.0605471810146823) q[2];
cx q[1],q[2];
ry(1.3452010353060837) q[1];
ry(1.0780593893504329) q[2];
cx q[1],q[2];
ry(-2.8292403519764693) q[2];
ry(2.077722119541467) q[5];
cx q[2],q[5];
ry(1.9572087570966192) q[2];
ry(-2.248327306058319) q[5];
cx q[2],q[5];
ry(1.8080187627259172) q[3];
ry(-1.553185620698238) q[4];
cx q[3],q[4];
ry(0.3586912222265791) q[3];
ry(1.1700946783749098) q[4];
cx q[3],q[4];
ry(1.3802052486101335) q[4];
ry(2.599354267947543) q[7];
cx q[4],q[7];
ry(0.012600442681343955) q[4];
ry(-0.10536779877719707) q[7];
cx q[4],q[7];
ry(-0.24708542173026002) q[5];
ry(-0.5123190033335545) q[6];
cx q[5],q[6];
ry(-0.03837692945948312) q[5];
ry(0.026997785938183125) q[6];
cx q[5],q[6];
ry(2.731502053398643) q[6];
ry(0.3853063857464534) q[9];
cx q[6],q[9];
ry(-3.1141293075667242) q[6];
ry(3.096990443225882) q[9];
cx q[6],q[9];
ry(-1.575974267884468) q[7];
ry(1.4957614365927716) q[8];
cx q[7],q[8];
ry(-0.8252482029799878) q[7];
ry(0.4432171443251667) q[8];
cx q[7],q[8];
ry(2.1424041767328816) q[8];
ry(2.960163855166252) q[11];
cx q[8],q[11];
ry(0.8536903798516517) q[8];
ry(0.15491264499144727) q[11];
cx q[8],q[11];
ry(0.47511354205984) q[9];
ry(0.2945783939205667) q[10];
cx q[9],q[10];
ry(2.0175000841411888) q[9];
ry(-2.573647955177509) q[10];
cx q[9],q[10];
ry(0.6679853786906359) q[0];
ry(-2.6678838375670018) q[1];
cx q[0],q[1];
ry(-0.43816459510450034) q[0];
ry(-1.3892941144224702) q[1];
cx q[0],q[1];
ry(1.1743251534891908) q[2];
ry(-0.051102953112043714) q[3];
cx q[2],q[3];
ry(2.703304684852481) q[2];
ry(1.1517750406653482) q[3];
cx q[2],q[3];
ry(2.5343451778752955) q[4];
ry(-0.5256368895436747) q[5];
cx q[4],q[5];
ry(-2.6625468915266826) q[4];
ry(-1.178283716394307) q[5];
cx q[4],q[5];
ry(0.7326214968352103) q[6];
ry(-1.6000049115698298) q[7];
cx q[6],q[7];
ry(-2.8708881252744884) q[6];
ry(1.5605085977418325) q[7];
cx q[6],q[7];
ry(2.4624239508178305) q[8];
ry(-2.583583381198124) q[9];
cx q[8],q[9];
ry(3.1255460043510186) q[8];
ry(0.1515412766886479) q[9];
cx q[8],q[9];
ry(-2.785385357967431) q[10];
ry(-2.4095643369690976) q[11];
cx q[10],q[11];
ry(0.09911295315510404) q[10];
ry(-0.9919971333914317) q[11];
cx q[10],q[11];
ry(-1.1074232167175904) q[0];
ry(2.3228375760717497) q[2];
cx q[0],q[2];
ry(2.212376651724922) q[0];
ry(-2.43856722958857) q[2];
cx q[0],q[2];
ry(3.0312452543569424) q[2];
ry(-1.95213759390658) q[4];
cx q[2],q[4];
ry(-2.870989976348607) q[2];
ry(1.140472038846931) q[4];
cx q[2],q[4];
ry(-2.0730222946947787) q[4];
ry(-1.2626272566648344) q[6];
cx q[4],q[6];
ry(3.1236531647964405) q[4];
ry(-3.122220960505411) q[6];
cx q[4],q[6];
ry(-0.6770126226370934) q[6];
ry(-1.2374288355074683) q[8];
cx q[6],q[8];
ry(-1.5373114361026163) q[6];
ry(-3.1142541474733183) q[8];
cx q[6],q[8];
ry(3.0183204305481044) q[8];
ry(-2.135816487875778) q[10];
cx q[8],q[10];
ry(-2.7505899600315598) q[8];
ry(0.09324422561736241) q[10];
cx q[8],q[10];
ry(-1.9258553504701488) q[1];
ry(1.2821681398175406) q[3];
cx q[1],q[3];
ry(0.11255567874423723) q[1];
ry(2.72497955776607) q[3];
cx q[1],q[3];
ry(1.9897490455723115) q[3];
ry(2.001947570022641) q[5];
cx q[3],q[5];
ry(0.7476924116744081) q[3];
ry(1.2687055389481694) q[5];
cx q[3],q[5];
ry(-2.920038490513485) q[5];
ry(0.9069824367335642) q[7];
cx q[5],q[7];
ry(0.0043855291830494645) q[5];
ry(0.016057912972099686) q[7];
cx q[5],q[7];
ry(-1.1196016934648934) q[7];
ry(-1.467809258239054) q[9];
cx q[7],q[9];
ry(1.189832812204419) q[7];
ry(3.032880355484515) q[9];
cx q[7],q[9];
ry(0.07647306908522644) q[9];
ry(-1.8034438112159012) q[11];
cx q[9],q[11];
ry(1.4419022667722792) q[9];
ry(1.55385377080384) q[11];
cx q[9],q[11];
ry(-1.0586477265619285) q[0];
ry(0.3760414047882721) q[3];
cx q[0],q[3];
ry(-0.9355792869324011) q[0];
ry(-0.8468260653383727) q[3];
cx q[0],q[3];
ry(-0.30937487857980756) q[1];
ry(2.2544543239465478) q[2];
cx q[1],q[2];
ry(-0.6059923350548928) q[1];
ry(2.3821283824879287) q[2];
cx q[1],q[2];
ry(0.14739125819198098) q[2];
ry(0.17221034137358338) q[5];
cx q[2],q[5];
ry(-1.6891704928145694) q[2];
ry(-1.469961722402982) q[5];
cx q[2],q[5];
ry(-0.5304337705637795) q[3];
ry(-0.7310939500829158) q[4];
cx q[3],q[4];
ry(1.5586997197483434) q[3];
ry(1.5705240527065811) q[4];
cx q[3],q[4];
ry(0.0854739307543301) q[4];
ry(1.5023849362983803) q[7];
cx q[4],q[7];
ry(-0.00946015570367198) q[4];
ry(3.135215919862865) q[7];
cx q[4],q[7];
ry(-0.10991880611697488) q[5];
ry(2.9833849024411294) q[6];
cx q[5],q[6];
ry(-3.1343447773879776) q[5];
ry(-0.027700666512478378) q[6];
cx q[5],q[6];
ry(-2.529698704357837) q[6];
ry(-2.487534596840137) q[9];
cx q[6],q[9];
ry(1.8044775910294852) q[6];
ry(3.133164931614446) q[9];
cx q[6],q[9];
ry(-0.3045308947746587) q[7];
ry(1.4620087975756269) q[8];
cx q[7],q[8];
ry(-0.5455246592590511) q[7];
ry(3.019251439050261) q[8];
cx q[7],q[8];
ry(-0.004305075029953364) q[8];
ry(-1.6613559352306009) q[11];
cx q[8],q[11];
ry(-1.5505031698301384) q[8];
ry(1.6362570236044018) q[11];
cx q[8],q[11];
ry(0.36838378545558026) q[9];
ry(-1.3437891561236393) q[10];
cx q[9],q[10];
ry(-0.8191891538753557) q[9];
ry(-2.2065981230414655) q[10];
cx q[9],q[10];
ry(2.705702501976016) q[0];
ry(2.6329804613799546) q[1];
cx q[0],q[1];
ry(-1.5257186927119852) q[0];
ry(1.6761745924726157) q[1];
cx q[0],q[1];
ry(-1.5360422717534867) q[2];
ry(-1.0711631597327038) q[3];
cx q[2],q[3];
ry(-1.5039524223020555) q[2];
ry(-1.164485603472979) q[3];
cx q[2],q[3];
ry(2.0826589462014127) q[4];
ry(2.741725871136225) q[5];
cx q[4],q[5];
ry(1.194321029529243) q[4];
ry(-2.5536817697954297) q[5];
cx q[4],q[5];
ry(-2.3464236134954155) q[6];
ry(2.952549275819178) q[7];
cx q[6],q[7];
ry(1.5139388457551952) q[6];
ry(-0.5478283965449346) q[7];
cx q[6],q[7];
ry(-1.672304447119035) q[8];
ry(0.7374138061320433) q[9];
cx q[8],q[9];
ry(2.5874358402858397) q[8];
ry(0.1705914013260435) q[9];
cx q[8],q[9];
ry(2.0629274212484807) q[10];
ry(1.6694108389561382) q[11];
cx q[10],q[11];
ry(-0.06592032988274354) q[10];
ry(-0.5478864290060432) q[11];
cx q[10],q[11];
ry(-1.5296772976350956) q[0];
ry(-1.7572250368261022) q[2];
cx q[0],q[2];
ry(1.882219166448343) q[0];
ry(-2.573752390983441) q[2];
cx q[0],q[2];
ry(-2.841040065562453) q[2];
ry(3.0178444609564) q[4];
cx q[2],q[4];
ry(1.4686617082770121) q[2];
ry(-3.022681221419882) q[4];
cx q[2],q[4];
ry(1.6104752928147485) q[4];
ry(0.6900635511959528) q[6];
cx q[4],q[6];
ry(3.1382924591788623) q[4];
ry(3.125628158800777) q[6];
cx q[4],q[6];
ry(0.0581902600380651) q[6];
ry(2.3176903015896424) q[8];
cx q[6],q[8];
ry(1.4780240339999096) q[6];
ry(2.905631852038654) q[8];
cx q[6],q[8];
ry(1.739298850836736) q[8];
ry(-0.900109800434204) q[10];
cx q[8],q[10];
ry(-0.14289393746586043) q[8];
ry(0.15463847223252117) q[10];
cx q[8],q[10];
ry(-0.08821920577900093) q[1];
ry(2.4986207244520897) q[3];
cx q[1],q[3];
ry(-1.914862952826283) q[1];
ry(-0.7470235116434276) q[3];
cx q[1],q[3];
ry(-1.3138415835279371) q[3];
ry(1.7755043277258045) q[5];
cx q[3],q[5];
ry(-0.9722940029384844) q[3];
ry(0.3299155675099197) q[5];
cx q[3],q[5];
ry(0.4765379231808762) q[5];
ry(-1.4703419571553575) q[7];
cx q[5],q[7];
ry(-1.585952563016561) q[5];
ry(-1.5507863622281128) q[7];
cx q[5],q[7];
ry(1.4311534491815694) q[7];
ry(1.9282821706596014) q[9];
cx q[7],q[9];
ry(3.1359701710640335) q[7];
ry(-0.02247580538330102) q[9];
cx q[7],q[9];
ry(-1.2175112434486441) q[9];
ry(-0.9537964764923047) q[11];
cx q[9],q[11];
ry(-1.4752136420516164) q[9];
ry(1.8756621139680953) q[11];
cx q[9],q[11];
ry(-0.33203302592607964) q[0];
ry(2.122713379154591) q[3];
cx q[0],q[3];
ry(1.936955940535419) q[0];
ry(-1.6398841034257396) q[3];
cx q[0],q[3];
ry(3.031774662133071) q[1];
ry(-0.3416289728881986) q[2];
cx q[1],q[2];
ry(0.6329561286026032) q[1];
ry(1.034970374317532) q[2];
cx q[1],q[2];
ry(0.9592209854403606) q[2];
ry(0.5590340281403119) q[5];
cx q[2],q[5];
ry(-3.1330858556478685) q[2];
ry(3.137624409501719) q[5];
cx q[2],q[5];
ry(-1.6615423495525254) q[3];
ry(-0.29787308452704675) q[4];
cx q[3],q[4];
ry(-1.0323596382476383) q[3];
ry(-1.7719918441384273) q[4];
cx q[3],q[4];
ry(0.3511477629577411) q[4];
ry(-2.989407829767826) q[7];
cx q[4],q[7];
ry(1.0399152777256757) q[4];
ry(-3.13792952422408) q[7];
cx q[4],q[7];
ry(-1.117263811175678) q[5];
ry(-0.6774304902069589) q[6];
cx q[5],q[6];
ry(3.139505990732174) q[5];
ry(3.0555224890069645) q[6];
cx q[5],q[6];
ry(-2.4278056117123707) q[6];
ry(1.2460397871012237) q[9];
cx q[6],q[9];
ry(0.05542792499553073) q[6];
ry(2.9426276221028975) q[9];
cx q[6],q[9];
ry(-3.0632736069816344) q[7];
ry(0.6912041365401472) q[8];
cx q[7],q[8];
ry(1.5701666412621051) q[7];
ry(-0.0035301064558783683) q[8];
cx q[7],q[8];
ry(1.5685574536959155) q[8];
ry(1.5737857783421778) q[11];
cx q[8],q[11];
ry(1.573782810638792) q[8];
ry(-1.5614912838738446) q[11];
cx q[8],q[11];
ry(2.5161974171440935) q[9];
ry(-0.05754854354562333) q[10];
cx q[9],q[10];
ry(-2.3206053260898716) q[9];
ry(1.5833430780374593) q[10];
cx q[9],q[10];
ry(-0.5529974569687369) q[0];
ry(1.076843550589703) q[1];
cx q[0],q[1];
ry(0.6071909601873716) q[0];
ry(0.9320946347066609) q[1];
cx q[0],q[1];
ry(2.0001931837736278) q[2];
ry(2.0998308027769745) q[3];
cx q[2],q[3];
ry(1.2137771722283919) q[2];
ry(0.8178393268191613) q[3];
cx q[2],q[3];
ry(1.502262470347344) q[4];
ry(-0.7066274151350616) q[5];
cx q[4],q[5];
ry(3.130741594537213) q[4];
ry(3.135883438593744) q[5];
cx q[4],q[5];
ry(0.32197504043419034) q[6];
ry(1.897653506965244) q[7];
cx q[6],q[7];
ry(-0.00401002403278594) q[6];
ry(-0.0059714294053234696) q[7];
cx q[6],q[7];
ry(-2.1010737086569136) q[8];
ry(-0.08619076704800016) q[9];
cx q[8],q[9];
ry(1.5720373797565192) q[8];
ry(-3.1131657408690843) q[9];
cx q[8],q[9];
ry(2.814262142593851) q[10];
ry(0.005085896401847201) q[11];
cx q[10],q[11];
ry(1.142042979215125) q[10];
ry(1.5707978891649077) q[11];
cx q[10],q[11];
ry(-0.8617601173335371) q[0];
ry(-2.5037675230714913) q[2];
cx q[0],q[2];
ry(-0.33225440175074245) q[0];
ry(1.20261697273889) q[2];
cx q[0],q[2];
ry(-2.0895001630963645) q[2];
ry(-2.023324656316543) q[4];
cx q[2],q[4];
ry(-1.5222865334552833) q[2];
ry(1.814954923005116) q[4];
cx q[2],q[4];
ry(0.4582894914360711) q[4];
ry(1.4102992686023468) q[6];
cx q[4],q[6];
ry(0.006517734237014028) q[4];
ry(-0.004106046692335193) q[6];
cx q[4],q[6];
ry(-0.9038331649769162) q[6];
ry(-2.105083640954379) q[8];
cx q[6],q[8];
ry(1.5796990526086738) q[6];
ry(3.1395164991193014) q[8];
cx q[6],q[8];
ry(-1.494292977275899) q[8];
ry(3.137184410218566) q[10];
cx q[8],q[10];
ry(1.5703147095244505) q[8];
ry(3.111616360247524) q[10];
cx q[8],q[10];
ry(-2.363920711403144) q[1];
ry(1.6020924897889741) q[3];
cx q[1],q[3];
ry(2.898143340230347) q[1];
ry(1.8866604961640627) q[3];
cx q[1],q[3];
ry(1.3008912955580296) q[3];
ry(0.647920953868375) q[5];
cx q[3],q[5];
ry(0.002044228318092145) q[3];
ry(-3.136574550597039) q[5];
cx q[3],q[5];
ry(-2.832451368616703) q[5];
ry(-0.4537625695490286) q[7];
cx q[5],q[7];
ry(-2.9425805672147693) q[5];
ry(-1.54686826344847) q[7];
cx q[5],q[7];
ry(1.5479222625608506) q[7];
ry(1.5613365952122964) q[9];
cx q[7],q[9];
ry(3.1279152758929687) q[7];
ry(2.395677311700432) q[9];
cx q[7],q[9];
ry(-1.5651791447990622) q[9];
ry(0.004419228791330276) q[11];
cx q[9],q[11];
ry(-1.4869766951273777) q[9];
ry(-2.9991705049726676) q[11];
cx q[9],q[11];
ry(-1.435468469254197) q[0];
ry(1.5714382230473993) q[3];
cx q[0],q[3];
ry(1.8996271908740807) q[0];
ry(1.3055008246049011) q[3];
cx q[0],q[3];
ry(0.4900939062025784) q[1];
ry(2.6974265191270295) q[2];
cx q[1],q[2];
ry(-1.6744319342866998) q[1];
ry(2.027629104137796) q[2];
cx q[1],q[2];
ry(-1.1447847000755438) q[2];
ry(-0.9852043198263587) q[5];
cx q[2],q[5];
ry(0.013783430511419859) q[2];
ry(-0.1147361153267513) q[5];
cx q[2],q[5];
ry(-2.019157210258629) q[3];
ry(-0.5501775950356178) q[4];
cx q[3],q[4];
ry(1.2070081869682312) q[3];
ry(-1.967571517268823) q[4];
cx q[3],q[4];
ry(-0.36198590380411116) q[4];
ry(1.4688902589463595) q[7];
cx q[4],q[7];
ry(3.132033356480627) q[4];
ry(0.002579990093714457) q[7];
cx q[4],q[7];
ry(1.622924668089083) q[5];
ry(0.842774121670482) q[6];
cx q[5],q[6];
ry(0.023560482418697248) q[5];
ry(1.5978728750875684) q[6];
cx q[5],q[6];
ry(-3.000782148487918) q[6];
ry(-1.5564597049692885) q[9];
cx q[6],q[9];
ry(-1.577855317139898) q[6];
ry(0.0006969229505564556) q[9];
cx q[6],q[9];
ry(-0.10642636175634924) q[7];
ry(-1.992352467949372) q[8];
cx q[7],q[8];
ry(0.0015635121816453628) q[7];
ry(3.0733829103277186) q[8];
cx q[7],q[8];
ry(-2.097349352228865) q[8];
ry(-3.140395490168662) q[11];
cx q[8],q[11];
ry(1.5734639946655733) q[8];
ry(0.006090280787265812) q[11];
cx q[8],q[11];
ry(-0.08573025671382797) q[9];
ry(-0.0734875842375248) q[10];
cx q[9],q[10];
ry(1.5757627655195574) q[9];
ry(1.5707866699258046) q[10];
cx q[9],q[10];
ry(-2.7080226601008377) q[0];
ry(1.2496400318449061) q[1];
ry(-1.8738590218759377) q[2];
ry(-0.3281165962632899) q[3];
ry(1.6932823110014361) q[4];
ry(-1.557750950899309) q[5];
ry(0.6598498628494216) q[6];
ry(1.5812710720209333) q[7];
ry(1.0269425857304346) q[8];
ry(1.572575065931741) q[9];
ry(-0.005708050307696943) q[10];
ry(1.6082898957708096) q[11];