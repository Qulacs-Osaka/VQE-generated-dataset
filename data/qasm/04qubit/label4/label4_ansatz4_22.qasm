OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.9175595492413091) q[0];
rz(-0.03466342526582533) q[0];
ry(2.007265240544025) q[1];
rz(-0.787181145772899) q[1];
ry(1.2366348526709112) q[2];
rz(-2.566773501539798) q[2];
ry(1.5342036983412806) q[3];
rz(-2.914155456862169) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.2965641877047176) q[0];
rz(-2.2987239293632653) q[0];
ry(-0.03180153838256037) q[1];
rz(-2.8620933834058655) q[1];
ry(-2.031753009746658) q[2];
rz(0.7230711387551407) q[2];
ry(-1.7534463735561356) q[3];
rz(-2.462414742162083) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.115274451993152) q[0];
rz(1.0011899011621601) q[0];
ry(0.9021286982170817) q[1];
rz(-1.9196298946071426) q[1];
ry(2.1260052764600843) q[2];
rz(2.306081525982658) q[2];
ry(-0.5809294851076627) q[3];
rz(-2.734193189412298) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.812877229341548) q[0];
rz(-1.0219512277191871) q[0];
ry(2.9238680399030024) q[1];
rz(-0.30555600956260776) q[1];
ry(0.27739517819839055) q[2];
rz(-1.691032868835561) q[2];
ry(0.20506316729846447) q[3];
rz(-0.8291454163676173) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.2994281775083962) q[0];
rz(-1.1066645851955714) q[0];
ry(-0.076445372371305) q[1];
rz(0.13361220701496943) q[1];
ry(0.10770584928782755) q[2];
rz(-1.1529979968274624) q[2];
ry(-2.531166669959482) q[3];
rz(-1.4734993079334853) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4198441928112686) q[0];
rz(0.4319641281201205) q[0];
ry(1.6767487002191963) q[1];
rz(0.30709284274830834) q[1];
ry(2.7623596217712794) q[2];
rz(0.5676818591741939) q[2];
ry(1.7504552989857896) q[3];
rz(3.0856903992892355) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.9023422632234954) q[0];
rz(-2.5091086856631524) q[0];
ry(2.786075795713321) q[1];
rz(1.719781688695863) q[1];
ry(-0.524420661083558) q[2];
rz(-1.4853866446509258) q[2];
ry(2.7583603465766737) q[3];
rz(-2.3213004091046003) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.9852260159757829) q[0];
rz(0.8410917261882126) q[0];
ry(-0.9201473319935429) q[1];
rz(2.095671346110305) q[1];
ry(2.714829139270679) q[2];
rz(0.7495750240405998) q[2];
ry(2.747991519037953) q[3];
rz(0.8825443227326585) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4214147130867723) q[0];
rz(1.3042542472518888) q[0];
ry(0.259252884861513) q[1];
rz(1.4841779192457272) q[1];
ry(-2.2407101811699333) q[2];
rz(0.7436418726601243) q[2];
ry(-1.2643354215634233) q[3];
rz(-1.9969893447809195) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1858019364914263) q[0];
rz(-2.886655023649233) q[0];
ry(-0.14747376969304682) q[1];
rz(2.116985595910487) q[1];
ry(-0.4334370099047939) q[2];
rz(-0.8639197177615863) q[2];
ry(-1.5753743318497462) q[3];
rz(-0.03224956974274029) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.13475533813665042) q[0];
rz(-0.6098730150592422) q[0];
ry(-2.327375260417924) q[1];
rz(1.6365639126976994) q[1];
ry(0.38050355137755076) q[2];
rz(-0.02991792793338633) q[2];
ry(-1.1073946497905744) q[3];
rz(-2.014213983930958) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.3898107228642207) q[0];
rz(-2.701292966898945) q[0];
ry(1.9740881634097425) q[1];
rz(2.445413753014547) q[1];
ry(1.9134214880812568) q[2];
rz(-0.08211790308718939) q[2];
ry(-1.8651127668068652) q[3];
rz(0.5044836572677092) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5766037640209776) q[0];
rz(-0.7875577117987463) q[0];
ry(-1.278992596931609) q[1];
rz(2.16678289392829) q[1];
ry(2.6665942519876564) q[2];
rz(-2.6066945261056507) q[2];
ry(1.3599696902758116) q[3];
rz(-1.56569295085838) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.8098758602317915) q[0];
rz(3.0140644997244657) q[0];
ry(0.19568314669423892) q[1];
rz(-0.5128259744400074) q[1];
ry(-1.443206077687262) q[2];
rz(0.3991817294776033) q[2];
ry(-2.4572821134750513) q[3];
rz(-1.3911110016074921) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.8838612009348428) q[0];
rz(-1.7573243899459259) q[0];
ry(0.23852599666542318) q[1];
rz(-1.2217164775142395) q[1];
ry(2.8974831459952264) q[2];
rz(-0.6568271121486368) q[2];
ry(-0.7083789165672014) q[3];
rz(-2.9339705900392774) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9828987062674105) q[0];
rz(2.726323513000548) q[0];
ry(-0.22031266642971303) q[1];
rz(1.1211479333859682) q[1];
ry(0.9793653695666026) q[2];
rz(2.358366797993654) q[2];
ry(0.01680881288190349) q[3];
rz(-2.8186304111746163) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5865137476459488) q[0];
rz(-2.4474374762009963) q[0];
ry(-0.6577080275739124) q[1];
rz(0.7106777903433165) q[1];
ry(0.15806777653827098) q[2];
rz(2.9558053422571784) q[2];
ry(-2.9291187063340334) q[3];
rz(-1.3109289264216173) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.1573139242710644) q[0];
rz(2.2784203549312227) q[0];
ry(2.8494535367508496) q[1];
rz(-1.9715662006461985) q[1];
ry(2.101163105145792) q[2];
rz(-1.6752767117431102) q[2];
ry(-0.17551690630647876) q[3];
rz(2.260313038622579) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.2158499169860981) q[0];
rz(-0.19930637025009704) q[0];
ry(-1.4248455660624968) q[1];
rz(-2.792281708114434) q[1];
ry(-0.045244055727343024) q[2];
rz(2.2957304696137326) q[2];
ry(2.382729462366792) q[3];
rz(-2.1205221772003746) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3737854138480774) q[0];
rz(2.072039243707081) q[0];
ry(2.923836099136591) q[1];
rz(-0.18859145015679157) q[1];
ry(-0.7789191724128484) q[2];
rz(0.3690010094095455) q[2];
ry(1.0284622022289378) q[3];
rz(2.918073013524034) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.4655351159739656) q[0];
rz(-2.9587948081707043) q[0];
ry(-0.5105528912494446) q[1];
rz(1.605875718581505) q[1];
ry(1.643759080627908) q[2];
rz(-2.2481533065427453) q[2];
ry(-1.8879689166421034) q[3];
rz(-0.3186706658180172) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7253386884802193) q[0];
rz(2.4319150543159074) q[0];
ry(0.005319957074314429) q[1];
rz(2.4156769099880906) q[1];
ry(-3.042355623759988) q[2];
rz(2.858200899358869) q[2];
ry(2.869236430883119) q[3];
rz(-1.07589016912834) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(3.0530062362561963) q[0];
rz(-1.77481435767765) q[0];
ry(1.2223704144840788) q[1];
rz(0.8126435890796451) q[1];
ry(-1.854514701860949) q[2];
rz(-1.9408542435495002) q[2];
ry(-2.3724833062999613) q[3];
rz(-3.12952968293446) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.6433515144328279) q[0];
rz(-0.08837212451078004) q[0];
ry(-2.94351995781407) q[1];
rz(-2.96605582884156) q[1];
ry(0.9346748524753278) q[2];
rz(0.9367731386374185) q[2];
ry(-1.614271393278259) q[3];
rz(1.3021284818118586) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.076577512019659) q[0];
rz(-2.2868103719647417) q[0];
ry(-1.1337053885989263) q[1];
rz(0.26265430642649257) q[1];
ry(-1.113544258551876) q[2];
rz(2.931690154320364) q[2];
ry(-1.4200778714650522) q[3];
rz(1.5818636504677983) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9032487113413232) q[0];
rz(0.0545229041068366) q[0];
ry(1.615605674688469) q[1];
rz(1.2501673147780545) q[1];
ry(-0.917723977490259) q[2];
rz(2.2418965577502306) q[2];
ry(-0.63000140525672) q[3];
rz(-0.39874163408588187) q[3];