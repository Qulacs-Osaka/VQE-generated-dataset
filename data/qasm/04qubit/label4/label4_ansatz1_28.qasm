OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.785857686437855) q[0];
rz(1.055505540299723) q[0];
ry(-0.4342047667894623) q[1];
rz(0.7109311335056541) q[1];
ry(3.132699290885379) q[2];
rz(1.0490430881353197) q[2];
ry(-0.18991744964227858) q[3];
rz(0.8757240168323449) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.105305627539384) q[0];
rz(-1.8912445792740393) q[0];
ry(-1.445915105028049) q[1];
rz(3.0325195354985848) q[1];
ry(-1.7979201380227277) q[2];
rz(0.58842511761195) q[2];
ry(1.7429334100041098) q[3];
rz(1.0040041126678316) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.38905278787527436) q[0];
rz(-0.6511543478723756) q[0];
ry(-0.9121691534880939) q[1];
rz(2.527014529580241) q[1];
ry(-3.0558388568346366) q[2];
rz(-1.3043036813071278) q[2];
ry(3.095555070697712) q[3];
rz(-0.8348066608719145) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.61213954332674) q[0];
rz(-0.9944190958660017) q[0];
ry(-1.0066076751803195) q[1];
rz(-0.5330048475267484) q[1];
ry(-0.7341363171826438) q[2];
rz(-1.1055213440343004) q[2];
ry(3.0718509988317786) q[3];
rz(-1.996535346553414) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.082141619918834) q[0];
rz(-1.4063465943311986) q[0];
ry(2.396371063830684) q[1];
rz(1.3610887994265664) q[1];
ry(-1.0844029826019836) q[2];
rz(-1.9093672926357517) q[2];
ry(-2.8439297931560414) q[3];
rz(1.4888148249678084) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.228159235226296) q[0];
rz(2.499344451349423) q[0];
ry(2.1102054008192255) q[1];
rz(1.9232369458976255) q[1];
ry(-0.20440502695161644) q[2];
rz(2.4423792413068948) q[2];
ry(0.7440225999401713) q[3];
rz(-1.0792942591032486) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.40730472512782606) q[0];
rz(-2.3915630123634357) q[0];
ry(1.8972025731462945) q[1];
rz(0.016019791437453357) q[1];
ry(-1.8942973501867604) q[2];
rz(1.4734079011284225) q[2];
ry(-0.2810383985797422) q[3];
rz(-1.27165087940224) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.6846935347916379) q[0];
rz(0.6481822425771778) q[0];
ry(1.549615330141875) q[1];
rz(2.250038755041999) q[1];
ry(-2.6168649878322086) q[2];
rz(0.6942089768343943) q[2];
ry(1.6832035172782298) q[3];
rz(1.2129360481105185) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1035409341799856) q[0];
rz(2.0480004593117007) q[0];
ry(3.0340561404587563) q[1];
rz(-2.6629536259022384) q[1];
ry(1.1385203411271922) q[2];
rz(-2.7714757190169412) q[2];
ry(2.5286548653695133) q[3];
rz(2.370034205911744) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.1420434472020777) q[0];
rz(0.20764457686936033) q[0];
ry(1.3949977853540316) q[1];
rz(1.0723139648534457) q[1];
ry(1.8939747415565247) q[2];
rz(2.041795336548925) q[2];
ry(0.4045424421626116) q[3];
rz(0.44771334046729633) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.4720183627042402) q[0];
rz(-2.881185730133686) q[0];
ry(-2.7090481925929044) q[1];
rz(-2.564114817588836) q[1];
ry(-1.0897043098023413) q[2];
rz(-0.3677872747876674) q[2];
ry(1.9001994533387134) q[3];
rz(2.342773702699817) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.2854513133685348) q[0];
rz(-0.6704295868298047) q[0];
ry(0.5547840639160488) q[1];
rz(-1.5358675847681347) q[1];
ry(3.1096919445253555) q[2];
rz(-2.6116579575515653) q[2];
ry(0.5452936804714517) q[3];
rz(2.573566211459709) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6248620004929524) q[0];
rz(-0.18514585785770704) q[0];
ry(-2.1988236224958664) q[1];
rz(-2.5165960605328563) q[1];
ry(-0.9963002792919422) q[2];
rz(3.0457662636714207) q[2];
ry(1.538367064043206) q[3];
rz(-1.1942278910297919) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.035097817446507) q[0];
rz(-2.480731649798781) q[0];
ry(1.7131450615006267) q[1];
rz(-3.1129732936475194) q[1];
ry(-0.6664755481757035) q[2];
rz(1.2930654352213775) q[2];
ry(2.1480223709927615) q[3];
rz(2.3594321742996214) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5905373827123315) q[0];
rz(2.2933420733200407) q[0];
ry(-2.8459382724012983) q[1];
rz(-1.2577980096971275) q[1];
ry(-1.8827937991930979) q[2];
rz(2.156557203813661) q[2];
ry(2.229025188663676) q[3];
rz(-0.42467644358301215) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.7228807303629643) q[0];
rz(-0.21914915049803102) q[0];
ry(1.1276530628147203) q[1];
rz(2.357599472080275) q[1];
ry(0.28002184975821187) q[2];
rz(-2.637845166100261) q[2];
ry(-2.356346871306817) q[3];
rz(2.959858273066847) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.5645914126491305) q[0];
rz(2.4601218942235383) q[0];
ry(-0.3792619016734287) q[1];
rz(-1.9395770819585252) q[1];
ry(0.750889313124846) q[2];
rz(0.7616658941038306) q[2];
ry(1.4054565174929952) q[3];
rz(1.52975667935069) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.6176217497169842) q[0];
rz(-1.5351837768767895) q[0];
ry(-0.6425552206265488) q[1];
rz(0.544056769222638) q[1];
ry(1.4772159920837706) q[2];
rz(2.795822182572176) q[2];
ry(0.6449765263358493) q[3];
rz(-2.283288506255722) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.715622290887766) q[0];
rz(0.7624735002867737) q[0];
ry(1.4307784808892023) q[1];
rz(2.1830102260662403) q[1];
ry(1.3776202034656162) q[2];
rz(-0.8161445853026531) q[2];
ry(0.7846772636180277) q[3];
rz(-0.2198970732301122) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4749522624319393) q[0];
rz(-2.0085310733076955) q[0];
ry(-2.491517504471955) q[1];
rz(2.1978195230594886) q[1];
ry(-0.7545188483899193) q[2];
rz(1.6247283445429612) q[2];
ry(2.70412801128387) q[3];
rz(-0.752475707532902) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.23778038218463207) q[0];
rz(-2.600318547702048) q[0];
ry(-1.3877560744935318) q[1];
rz(-1.028521572610078) q[1];
ry(-2.0498953289729567) q[2];
rz(-1.3963621038554734) q[2];
ry(-2.530516172211028) q[3];
rz(2.6225492805832364) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.8883250425770375) q[0];
rz(-1.5392398467108295) q[0];
ry(-0.8956294349274891) q[1];
rz(0.6856940573031309) q[1];
ry(2.8856648666969447) q[2];
rz(-2.834675901898296) q[2];
ry(-1.5532777390889825) q[3];
rz(-1.6223114921335224) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.0412693734164784) q[0];
rz(2.2883971266740617) q[0];
ry(1.2538157595277726) q[1];
rz(-2.631000782153982) q[1];
ry(0.9013823831809873) q[2];
rz(2.3151237698506617) q[2];
ry(2.555245893377302) q[3];
rz(-1.6020027560858399) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.963802535846061) q[0];
rz(1.8050292346013441) q[0];
ry(-2.959256914245417) q[1];
rz(1.4880252519607189) q[1];
ry(0.4200804077334904) q[2];
rz(-1.1654791989212614) q[2];
ry(2.3512471942516227) q[3];
rz(-1.3558633989135849) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.31035604251648685) q[0];
rz(1.1721052950906983) q[0];
ry(0.8343227283284961) q[1];
rz(-2.7711695293688448) q[1];
ry(-1.1569225578643412) q[2];
rz(0.12448662541212217) q[2];
ry(0.04078574474799445) q[3];
rz(-1.5250216447458709) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.368050386089707) q[0];
rz(0.025240185412837235) q[0];
ry(-1.7425632207399318) q[1];
rz(2.962621435451392) q[1];
ry(0.5992660485704882) q[2];
rz(0.23487735457619685) q[2];
ry(1.239710929106013) q[3];
rz(0.08070618056377676) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.014986544649786) q[0];
rz(-0.4291733520822642) q[0];
ry(2.8258835469473964) q[1];
rz(1.4832183518445827) q[1];
ry(-0.9052120313268261) q[2];
rz(0.7504390268619271) q[2];
ry(-1.932201723370862) q[3];
rz(0.22270489156454776) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1597893480179553) q[0];
rz(-2.084389826942613) q[0];
ry(2.5481605421473246) q[1];
rz(1.1733555262621693) q[1];
ry(-1.9342725240115342) q[2];
rz(2.5413244662897445) q[2];
ry(2.492600815079189) q[3];
rz(1.6119789151239021) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5495324967274893) q[0];
rz(-2.5915017134099734) q[0];
ry(-1.1775184892919288) q[1];
rz(2.333380603844619) q[1];
ry(0.38449869481419113) q[2];
rz(3.0119547549769305) q[2];
ry(-1.8243292591591427) q[3];
rz(-0.5541625962358742) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.7529538141947407) q[0];
rz(-2.2813930111647713) q[0];
ry(0.601433611493098) q[1];
rz(-3.134997815003548) q[1];
ry(-1.7318153294429182) q[2];
rz(-2.8761808183014073) q[2];
ry(-2.6506309268745407) q[3];
rz(2.0084034954260033) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8731118039660446) q[0];
rz(-2.378165374035958) q[0];
ry(1.0376976243075404) q[1];
rz(0.5521295008111896) q[1];
ry(2.7217393391988702) q[2];
rz(0.9127830870139362) q[2];
ry(0.012375191810106045) q[3];
rz(1.5669745272786184) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.0867050876088635) q[0];
rz(-3.0162672485412867) q[0];
ry(-0.3201048018020236) q[1];
rz(2.0586966264729387) q[1];
ry(-1.6642291940114036) q[2];
rz(0.012445325560234368) q[2];
ry(0.5635962994831409) q[3];
rz(2.0004058463458843) q[3];