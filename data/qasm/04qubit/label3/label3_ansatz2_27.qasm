OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.663775288131477) q[0];
rz(0.09633545445654157) q[0];
ry(0.9977123932572319) q[1];
rz(0.7594343540018403) q[1];
ry(-0.33689043231857774) q[2];
rz(-0.16129875781381386) q[2];
ry(0.0644838109576102) q[3];
rz(0.43440468290397766) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.1595408605022641) q[0];
rz(-1.7083273438686453) q[0];
ry(1.8624404280300952) q[1];
rz(-1.954189443695603) q[1];
ry(0.7621863029234025) q[2];
rz(0.6394654581107552) q[2];
ry(3.0686045702340143) q[3];
rz(1.3898560036950354) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.333568957718395) q[0];
rz(-0.41280212849494885) q[0];
ry(0.5193336966898785) q[1];
rz(-2.2152532715573283) q[1];
ry(0.630388626528557) q[2];
rz(-1.2678747560330068) q[2];
ry(2.1069142658674718) q[3];
rz(2.020648985516982) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.052960778622239246) q[0];
rz(0.5594072574501228) q[0];
ry(2.2053708413951276) q[1];
rz(-0.17754320561935077) q[1];
ry(0.7155841430189566) q[2];
rz(0.1396650780794281) q[2];
ry(-2.730679154780346) q[3];
rz(-2.611850171702812) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.8421523855808699) q[0];
rz(-1.8826084504494403) q[0];
ry(-1.1979227699839852) q[1];
rz(3.0446886506777644) q[1];
ry(0.6944516411963244) q[2];
rz(-1.3904059817995407) q[2];
ry(2.771806407986576) q[3];
rz(1.2053382508301627) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.00551602507630644) q[0];
rz(-0.028391344084497835) q[0];
ry(-1.7001635202072884) q[1];
rz(0.5590567309501404) q[1];
ry(0.3418312116110442) q[2];
rz(-0.023027083290487475) q[2];
ry(2.6497483276544083) q[3];
rz(-0.8058724015291688) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.6580902104577122) q[0];
rz(-0.558404982333105) q[0];
ry(-1.099548843846116) q[1];
rz(1.8438342063017947) q[1];
ry(2.750910258262128) q[2];
rz(-2.145135461549242) q[2];
ry(2.632647586152907) q[3];
rz(1.2109080112162895) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.9734565654297491) q[0];
rz(-2.86982068371352) q[0];
ry(2.5319402160818267) q[1];
rz(-1.712883052189607) q[1];
ry(2.644461068699363) q[2];
rz(0.32142462447686143) q[2];
ry(1.8550287985819658) q[3];
rz(0.08260750575853208) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(3.03106891198117) q[0];
rz(0.9636252941023339) q[0];
ry(-0.9061226313517233) q[1];
rz(-3.0744385692101655) q[1];
ry(2.1954652069246627) q[2];
rz(2.679241910318541) q[2];
ry(1.1344252519408022) q[3];
rz(0.8226506855403107) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.882672616929133) q[0];
rz(3.061809768606648) q[0];
ry(-2.6533507482309737) q[1];
rz(-2.8210574880590578) q[1];
ry(0.26492866533495013) q[2];
rz(-2.5926792236031093) q[2];
ry(-2.445725245858939) q[3];
rz(1.3603633452968191) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.7985636462962415) q[0];
rz(1.6508607856237472) q[0];
ry(-0.13875652312563838) q[1];
rz(-2.3230962905813524) q[1];
ry(-2.3469117875173215) q[2];
rz(-0.3299172138058104) q[2];
ry(-2.223527365549245) q[3];
rz(0.9475410614959126) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.737343664566433) q[0];
rz(-1.1835658178936832) q[0];
ry(-1.1626691958395006) q[1];
rz(-1.5563825191571627) q[1];
ry(1.7263166682782616) q[2];
rz(2.0121356932615386) q[2];
ry(2.4182737765205427) q[3];
rz(-1.8354008640096453) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.08194822460771786) q[0];
rz(-2.7669966537092567) q[0];
ry(-1.5386450974502353) q[1];
rz(-1.9033086916784892) q[1];
ry(-0.8194760587317131) q[2];
rz(-2.9346858701015512) q[2];
ry(-1.460726213406498) q[3];
rz(-1.231369322599589) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.0243236442998063) q[0];
rz(0.8369938289239157) q[0];
ry(-0.40892450552188114) q[1];
rz(-1.906708924701993) q[1];
ry(1.0699126108410524) q[2];
rz(-1.272152857004751) q[2];
ry(-0.7286475454826058) q[3];
rz(1.568058520030801) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.0279536281707635) q[0];
rz(0.739539925112769) q[0];
ry(1.286814187684097) q[1];
rz(-0.5955234702953814) q[1];
ry(-0.8652205540897882) q[2];
rz(-1.0240297127755198) q[2];
ry(2.1312381737058965) q[3];
rz(0.6972225685171888) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.7310487443898384) q[0];
rz(1.3166365243377967) q[0];
ry(3.0115861898448095) q[1];
rz(1.7733731449555288) q[1];
ry(0.737668740086582) q[2];
rz(-2.44355617672302) q[2];
ry(-0.4283001838623006) q[3];
rz(-1.119299078899746) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.6965638205634557) q[0];
rz(-1.9886887705100333) q[0];
ry(-0.8976446435096221) q[1];
rz(0.5793277158858405) q[1];
ry(1.8380795442087994) q[2];
rz(-1.0344251941890397) q[2];
ry(1.9443649916181192) q[3];
rz(0.060104343238347416) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.295187377698829) q[0];
rz(2.953826912633802) q[0];
ry(-1.8416203337549009) q[1];
rz(-2.4835118637682583) q[1];
ry(-0.9170988933340036) q[2];
rz(-2.156387594356418) q[2];
ry(-0.07696705699979574) q[3];
rz(0.561639480969931) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.4992641520078776) q[0];
rz(-1.6968360515249692) q[0];
ry(-1.4787838116709529) q[1];
rz(1.7839997108227157) q[1];
ry(1.9544367081923966) q[2];
rz(-2.487112896296651) q[2];
ry(-0.12245964402765576) q[3];
rz(1.76143668860162) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.8582277457950758) q[0];
rz(0.6831714321670821) q[0];
ry(-0.8752293937525041) q[1];
rz(-1.8234674448916417) q[1];
ry(-2.653336631711513) q[2];
rz(0.18020536027115863) q[2];
ry(0.23676415465729864) q[3];
rz(-2.16417312981753) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.6197317031325504) q[0];
rz(-2.1046600600842) q[0];
ry(-1.8657987001133831) q[1];
rz(-1.5448514703037768) q[1];
ry(-2.3301877462732374) q[2];
rz(-2.021503276087798) q[2];
ry(-1.7742917605063555) q[3];
rz(-1.3192921178220463) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.4235160670316207) q[0];
rz(1.8464284927400128) q[0];
ry(2.816355310393551) q[1];
rz(0.7810281427224556) q[1];
ry(-0.2747539653168882) q[2];
rz(2.4942357780430453) q[2];
ry(0.6930174572384784) q[3];
rz(-1.403436940273113) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.5764166691338826) q[0];
rz(2.5977224725109234) q[0];
ry(-0.3960752802092724) q[1];
rz(2.6038372503372353) q[1];
ry(0.05866523280339031) q[2];
rz(1.9455836492940055) q[2];
ry(-2.021499370165235) q[3];
rz(-2.6650891078576033) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.006030193197383) q[0];
rz(1.526623378919006) q[0];
ry(1.123450888086326) q[1];
rz(3.0285784031317378) q[1];
ry(-2.4107291776712976) q[2];
rz(0.03351213467742775) q[2];
ry(-1.7541553147160918) q[3];
rz(-2.4074107966342617) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.880396896715846) q[0];
rz(1.4471497954871986) q[0];
ry(1.2752857571529943) q[1];
rz(2.8743899068476724) q[1];
ry(2.9109693660483824) q[2];
rz(3.1340499555124133) q[2];
ry(0.27684910783135525) q[3];
rz(0.7262217892942192) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.7237666689531244) q[0];
rz(2.188816306645566) q[0];
ry(1.039264837474989) q[1];
rz(-2.24046581763519) q[1];
ry(-2.2266590159279227) q[2];
rz(-0.43151810158512227) q[2];
ry(0.5245822582570862) q[3];
rz(-0.39627421029983356) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.998253586400813) q[0];
rz(-1.922977345677456) q[0];
ry(-3.0581299792145136) q[1];
rz(-2.7740031107138625) q[1];
ry(-2.4250330848150288) q[2];
rz(-2.015716873200165) q[2];
ry(-1.3802534494173582) q[3];
rz(1.1642580067724984) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6681398394488152) q[0];
rz(1.1754537806083896) q[0];
ry(-1.2819708998464716) q[1];
rz(-1.037693733666225) q[1];
ry(1.7307315655107525) q[2];
rz(1.2074265020392048) q[2];
ry(2.0431624039385516) q[3];
rz(2.2249447029145233) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6748280294210813) q[0];
rz(1.4763518335036065) q[0];
ry(1.5525523171598632) q[1];
rz(0.7484466061613296) q[1];
ry(1.1667767622678191) q[2];
rz(-2.898276020580169) q[2];
ry(-1.083973194621743) q[3];
rz(0.3809196504886759) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6427672412654086) q[0];
rz(1.4051603689708154) q[0];
ry(2.7835764526151023) q[1];
rz(2.9895194611425215) q[1];
ry(1.5968052496085345) q[2];
rz(-0.8299173365265454) q[2];
ry(0.6174635089057192) q[3];
rz(2.7933663732526184) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.06176773162019705) q[0];
rz(-0.9028802329811917) q[0];
ry(-2.354837051375673) q[1];
rz(-0.45587530903657764) q[1];
ry(2.6557265669971755) q[2];
rz(-2.5069688734754325) q[2];
ry(-1.8148579264105331) q[3];
rz(-0.915720939932859) q[3];