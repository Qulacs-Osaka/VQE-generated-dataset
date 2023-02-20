OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.9169341266386644) q[0];
rz(2.613425344611721) q[0];
ry(2.567972168060565) q[1];
rz(0.45101486193684137) q[1];
ry(-2.5790552988733086) q[2];
rz(0.9504990976950102) q[2];
ry(-2.6830537821148) q[3];
rz(0.29884824253260567) q[3];
ry(-0.8828673956489093) q[4];
rz(0.16083736690976647) q[4];
ry(1.6444694625843228) q[5];
rz(-0.5201204197648659) q[5];
ry(-1.9670570995878247) q[6];
rz(2.45737071688001) q[6];
ry(2.889491930752369) q[7];
rz(2.4438589225084346) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.9888957140750364) q[0];
rz(1.1917202329239922) q[0];
ry(2.157431060575205) q[1];
rz(0.44837966725655015) q[1];
ry(0.8380441979631605) q[2];
rz(2.1303889137832392) q[2];
ry(-2.7449776725283668) q[3];
rz(1.5029254812488908) q[3];
ry(-1.585758312352528) q[4];
rz(-0.5637892941475626) q[4];
ry(0.7896028014659935) q[5];
rz(0.0105179958367618) q[5];
ry(1.3622960996553273) q[6];
rz(1.7345929509802733) q[6];
ry(-2.4526211036431653) q[7];
rz(-1.8801377950370535) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.390594180096374) q[0];
rz(2.1742689467702534) q[0];
ry(2.2463291706942012) q[1];
rz(-2.9814312922404347) q[1];
ry(2.1868830178365943) q[2];
rz(-2.141226050366014) q[2];
ry(1.3518693202212615) q[3];
rz(-1.4158065959474175) q[3];
ry(-1.1353119658148672) q[4];
rz(-1.713561985042006) q[4];
ry(-2.632466731506324) q[5];
rz(-2.3329372264003183) q[5];
ry(-0.1334761110991105) q[6];
rz(1.1902982326091616) q[6];
ry(-0.46631361109448194) q[7];
rz(-2.678004107123986) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.7254155619005004) q[0];
rz(-0.06476363013549276) q[0];
ry(-1.3510775035679188) q[1];
rz(-2.8355823748861027) q[1];
ry(0.7208144997042009) q[2];
rz(0.015127409245620059) q[2];
ry(-0.2990641174610804) q[3];
rz(-1.540374120055251) q[3];
ry(-2.9645874044015583) q[4];
rz(-2.4322654653219127) q[4];
ry(-0.5301323394114782) q[5];
rz(1.1882236522349687) q[5];
ry(2.637794412809946) q[6];
rz(-3.0076808657500744) q[6];
ry(-0.7467886459207655) q[7];
rz(-0.08990594274675859) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.6175562346077577) q[0];
rz(-2.815266028056343) q[0];
ry(-0.5764167868745795) q[1];
rz(1.474964691724499) q[1];
ry(-1.1667247683448534) q[2];
rz(-2.3971467130820026) q[2];
ry(-1.8234443896536794) q[3];
rz(-2.212898002678117) q[3];
ry(-1.2680783798609903) q[4];
rz(-1.89689423944474) q[4];
ry(0.8911350459564948) q[5];
rz(-0.7005347449589374) q[5];
ry(2.029706986161945) q[6];
rz(-2.4358266441283316) q[6];
ry(-2.5849058844465014) q[7];
rz(-1.6020901609183733) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.7292221204857883) q[0];
rz(-1.4189770588644848) q[0];
ry(-1.571062459977285) q[1];
rz(3.106452124708393) q[1];
ry(0.2587091277689493) q[2];
rz(3.0797251106885453) q[2];
ry(-2.3988120718777646) q[3];
rz(0.9451798867054225) q[3];
ry(-1.0208840413459015) q[4];
rz(-2.0243098961781394) q[4];
ry(-1.0595000568640758) q[5];
rz(-3.022546559120927) q[5];
ry(2.35368423789044) q[6];
rz(-2.3564185204516686) q[6];
ry(1.8646130830049479) q[7];
rz(-2.734487111693536) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.596166867903978) q[0];
rz(2.3836941207226148) q[0];
ry(1.3948824978760115) q[1];
rz(2.98810245736189) q[1];
ry(1.0265031125554396) q[2];
rz(-1.103562490068373) q[2];
ry(-1.1862943092045206) q[3];
rz(-1.042958776860937) q[3];
ry(-3.040117246657349) q[4];
rz(-1.9009091546210948) q[4];
ry(2.436310311909646) q[5];
rz(-0.8105934807742399) q[5];
ry(-2.5577549628529774) q[6];
rz(2.909026126002273) q[6];
ry(-1.1905289449603664) q[7];
rz(-0.2437655024214873) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(3.0864801138808198) q[0];
rz(2.620496606254051) q[0];
ry(-1.644206849807948) q[1];
rz(0.3913761292642493) q[1];
ry(-1.979410981383574) q[2];
rz(0.02315799695807108) q[2];
ry(-2.29887285616042) q[3];
rz(1.6738109849457405) q[3];
ry(-1.9113133757132301) q[4];
rz(-3.113824894413266) q[4];
ry(-1.787803993645073) q[5];
rz(1.5304561373104975) q[5];
ry(-0.4141806018632884) q[6];
rz(-0.1065288536190695) q[6];
ry(-1.6423108043169625) q[7];
rz(-0.5316464927557609) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.9271805246953768) q[0];
rz(-0.8855057459711322) q[0];
ry(-0.7583393013816817) q[1];
rz(0.050538791119552576) q[1];
ry(-0.7215206253117614) q[2];
rz(0.7495689757493815) q[2];
ry(-0.5689204922111024) q[3];
rz(2.3448325508937145) q[3];
ry(-2.4291283663166254) q[4];
rz(2.5907333316662613) q[4];
ry(-1.5372431664340411) q[5];
rz(-0.2292306449547324) q[5];
ry(2.8200207865276186) q[6];
rz(-1.0222501035332447) q[6];
ry(1.9798563591306015) q[7];
rz(-1.8497322001370442) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.4019386346721623) q[0];
rz(2.950506997474351) q[0];
ry(1.1117535800434604) q[1];
rz(2.393533044044588) q[1];
ry(-1.246109323316029) q[2];
rz(-2.8543530344823194) q[2];
ry(-0.919010455828719) q[3];
rz(-2.0374393347128517) q[3];
ry(1.1739809376839194) q[4];
rz(-2.798058670742693) q[4];
ry(-2.4435391838580918) q[5];
rz(-0.45585340105318384) q[5];
ry(0.29946757044837297) q[6];
rz(0.5962264913400349) q[6];
ry(1.4835125167270347) q[7];
rz(0.02937931794046023) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.7445235915536392) q[0];
rz(2.440175764005495) q[0];
ry(1.2371354081253862) q[1];
rz(-2.9762096354305467) q[1];
ry(1.1356341723561911) q[2];
rz(1.6276925128162432) q[2];
ry(-1.759246522265801) q[3];
rz(-2.0080797557875893) q[3];
ry(-2.469666637169271) q[4];
rz(-2.5652774116407238) q[4];
ry(0.9220903325772162) q[5];
rz(-0.264734751147774) q[5];
ry(-0.3758225066740853) q[6];
rz(2.7499912336373966) q[6];
ry(-0.5481520219398277) q[7];
rz(-2.1550765068232263) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.762819929957989) q[0];
rz(-2.643627352763383) q[0];
ry(1.6470119051999532) q[1];
rz(0.9011071042735193) q[1];
ry(1.2428245202963897) q[2];
rz(-1.431147402267098) q[2];
ry(-0.6868027115192707) q[3];
rz(-2.469826042969373) q[3];
ry(0.24180793164539338) q[4];
rz(0.5733012464005297) q[4];
ry(-1.0295778780769655) q[5];
rz(3.0126946063599807) q[5];
ry(-2.0248239894578384) q[6];
rz(-2.701425852598544) q[6];
ry(3.028752932442522) q[7];
rz(2.214969327784841) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.3157813049145324) q[0];
rz(2.5004884769445717) q[0];
ry(-0.26242103404184824) q[1];
rz(2.4165576594968203) q[1];
ry(-1.8370690478949196) q[2];
rz(2.2813534133133873) q[2];
ry(2.415878409072319) q[3];
rz(-0.931024208439004) q[3];
ry(1.7000914843752213) q[4];
rz(-0.26295296109679794) q[4];
ry(0.1096905158234387) q[5];
rz(-2.571367200883428) q[5];
ry(2.110631576436501) q[6];
rz(-0.21628711930743305) q[6];
ry(2.5495081362235807) q[7];
rz(-0.3724659237829142) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.9319705221109151) q[0];
rz(-3.0425518412695274) q[0];
ry(-2.546235100609505) q[1];
rz(-3.021658786451776) q[1];
ry(2.268385140355577) q[2];
rz(1.9243662467694644) q[2];
ry(-0.9739639317691674) q[3];
rz(-1.984793094753904) q[3];
ry(-1.3037118029294712) q[4];
rz(-0.6807911129835764) q[4];
ry(-0.14154537276815837) q[5];
rz(-2.0388211940431624) q[5];
ry(2.2199891893955424) q[6];
rz(-2.40907126358183) q[6];
ry(0.9368086520323562) q[7];
rz(0.049699585110897) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.9609048187518177) q[0];
rz(-0.9221466453035537) q[0];
ry(-0.059366147505122174) q[1];
rz(0.6035210200030292) q[1];
ry(-2.8016763763393753) q[2];
rz(0.00345282751805493) q[2];
ry(-2.2519385335870035) q[3];
rz(2.380472417170035) q[3];
ry(2.0012378430092603) q[4];
rz(-2.1504295479426565) q[4];
ry(-2.96115552511709) q[5];
rz(1.345242778133243) q[5];
ry(2.8631146598896895) q[6];
rz(-2.772302001777332) q[6];
ry(0.40406183177315763) q[7];
rz(0.8149844732251443) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.473055592134066) q[0];
rz(-2.9647852119077402) q[0];
ry(2.3694525837855593) q[1];
rz(0.673748477142629) q[1];
ry(-1.0389959843568688) q[2];
rz(2.607955158444485) q[2];
ry(-1.7405232258559695) q[3];
rz(2.3743670265966705) q[3];
ry(1.20802410019847) q[4];
rz(2.0221088576893935) q[4];
ry(1.0314869861603193) q[5];
rz(0.03664510639315014) q[5];
ry(-0.35147310490160033) q[6];
rz(2.1260920104402015) q[6];
ry(-0.9198823031407484) q[7];
rz(1.0554106886201005) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.8977580737713877) q[0];
rz(1.1714336618353751) q[0];
ry(1.5773874731753417) q[1];
rz(2.001455306832375) q[1];
ry(-2.055104660878301) q[2];
rz(-1.9450415066025146) q[2];
ry(-1.736464884985997) q[3];
rz(0.8286708164254339) q[3];
ry(0.8236821010926666) q[4];
rz(2.3094278459645667) q[4];
ry(0.10195722114376249) q[5];
rz(2.675330823122228) q[5];
ry(-2.225089794730054) q[6];
rz(-2.8594713958810782) q[6];
ry(1.265724207402008) q[7];
rz(-0.4506270591374806) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.5975516109271544) q[0];
rz(-0.5643375468853344) q[0];
ry(-0.29568055646907243) q[1];
rz(2.7152671392160697) q[1];
ry(1.3410348454436951) q[2];
rz(-0.863634524875053) q[2];
ry(-2.87848907464061) q[3];
rz(1.2960908603319918) q[3];
ry(-0.5366077065329237) q[4];
rz(1.3757388523383405) q[4];
ry(1.349610516912627) q[5];
rz(0.977558427932096) q[5];
ry(2.3114189902221995) q[6];
rz(1.9196082817445028) q[6];
ry(0.1423299173139508) q[7];
rz(1.7362238668108205) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.6857382663215756) q[0];
rz(2.8758472171523457) q[0];
ry(0.5812005723770433) q[1];
rz(1.271914117444196) q[1];
ry(2.858285955348887) q[2];
rz(-1.5658881503576632) q[2];
ry(1.6570911483578317) q[3];
rz(-2.189053528468375) q[3];
ry(-2.6902244372359254) q[4];
rz(-0.32211659074493504) q[4];
ry(-2.4386367859005516) q[5];
rz(1.1362095162169101) q[5];
ry(1.2877279017904357) q[6];
rz(-1.5416499765962375) q[6];
ry(0.13912933031470764) q[7];
rz(0.09869559048224819) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.50291576389917) q[0];
rz(-1.6381071357868917) q[0];
ry(-1.725819519393319) q[1];
rz(1.2239791886204872) q[1];
ry(-0.5191705372562903) q[2];
rz(2.11654840195719) q[2];
ry(-3.1177334470280313) q[3];
rz(-2.9001434643892177) q[3];
ry(0.650837389737692) q[4];
rz(-2.440204670241173) q[4];
ry(-1.913719602016427) q[5];
rz(-2.711404951604229) q[5];
ry(1.5822408358264948) q[6];
rz(2.7842697783588415) q[6];
ry(2.681459521974419) q[7];
rz(-1.9849418382245698) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.9234227075052811) q[0];
rz(-1.4494721953334908) q[0];
ry(-1.7809192069933586) q[1];
rz(-0.2458611745966861) q[1];
ry(2.5399537617533166) q[2];
rz(-1.2302072097719325) q[2];
ry(-1.480437460115672) q[3];
rz(2.0490516068066253) q[3];
ry(1.3418897659990936) q[4];
rz(-2.5881127904429473) q[4];
ry(-0.6232396196524164) q[5];
rz(-0.08176480394634834) q[5];
ry(-0.5800083810451425) q[6];
rz(-1.6898049965412916) q[6];
ry(1.522260814256553) q[7];
rz(2.7670744668078657) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.4432021504508481) q[0];
rz(1.8020063540052769) q[0];
ry(0.5827948795325986) q[1];
rz(1.9898165086117723) q[1];
ry(2.3081138714397538) q[2];
rz(-0.08887266904029101) q[2];
ry(-1.5860581118465582) q[3];
rz(2.7590035412946685) q[3];
ry(0.8735484089209603) q[4];
rz(-2.9180210948720084) q[4];
ry(1.9081471014722011) q[5];
rz(0.6037266430829976) q[5];
ry(0.562853092514243) q[6];
rz(3.140219327906015) q[6];
ry(-0.4641919843759404) q[7];
rz(2.295934285755215) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.131985324872309) q[0];
rz(1.9714027885477539) q[0];
ry(-2.8245535795754395) q[1];
rz(-1.1715556101349236) q[1];
ry(2.173271310120122) q[2];
rz(2.3397349158240845) q[2];
ry(1.6233858847126612) q[3];
rz(1.3085623665749622) q[3];
ry(1.444493726235009) q[4];
rz(-2.8165764519817706) q[4];
ry(-0.598040430190799) q[5];
rz(-1.0935182063780176) q[5];
ry(1.87591562929072) q[6];
rz(0.5624840203695288) q[6];
ry(-2.600165668348049) q[7];
rz(-0.7952016832582505) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.4310856935585785) q[0];
rz(-2.3411536280228766) q[0];
ry(-0.9908691685971629) q[1];
rz(0.5061409713220764) q[1];
ry(-1.5573967232350219) q[2];
rz(-1.6260097936160394) q[2];
ry(-2.5552791997178144) q[3];
rz(-1.7947224157598631) q[3];
ry(0.23210570443549638) q[4];
rz(0.44930667735552543) q[4];
ry(-0.5988043247690014) q[5];
rz(-0.49338946893734453) q[5];
ry(2.8497404346979334) q[6];
rz(-1.5071745695664163) q[6];
ry(2.66152846707335) q[7];
rz(2.376916223916271) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.08168642738741982) q[0];
rz(0.8857343582882535) q[0];
ry(1.3125097129819405) q[1];
rz(1.5254517878032234) q[1];
ry(-0.3588877393385571) q[2];
rz(-1.5989686634152867) q[2];
ry(0.22289190516372592) q[3];
rz(1.272371376279131) q[3];
ry(1.8331955031059466) q[4];
rz(0.6631951842402487) q[4];
ry(-2.128656414159903) q[5];
rz(-2.143491231634296) q[5];
ry(1.9986789786247776) q[6];
rz(-1.0592049687715082) q[6];
ry(1.3773837863330913) q[7];
rz(-2.626107295545582) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.7675217884231831) q[0];
rz(-0.7420517509489409) q[0];
ry(-1.8620417871473496) q[1];
rz(1.5790619179788423) q[1];
ry(1.5912914519378651) q[2];
rz(-1.7385451862463839) q[2];
ry(-3.029314072501582) q[3];
rz(1.368654754715147) q[3];
ry(-0.7848572810504869) q[4];
rz(-0.4789873346737661) q[4];
ry(-0.08572384494089319) q[5];
rz(0.16381869606515834) q[5];
ry(1.6509433218703964) q[6];
rz(-2.347994147107256) q[6];
ry(0.8980266013942163) q[7];
rz(-1.9184297640797683) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.435072436991609) q[0];
rz(-2.331711527950362) q[0];
ry(-0.7168614996264298) q[1];
rz(2.8371857914405734) q[1];
ry(0.7183594582090255) q[2];
rz(-1.0703573289455246) q[2];
ry(2.7272785691198704) q[3];
rz(1.82288153456992) q[3];
ry(1.3626433828494902) q[4];
rz(2.7244743391159028) q[4];
ry(1.6490787393472186) q[5];
rz(-2.4270162843132153) q[5];
ry(1.054334280606004) q[6];
rz(0.24267253300626412) q[6];
ry(-2.457699778710919) q[7];
rz(-0.29036268993000647) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.0166503273986738) q[0];
rz(1.9425725649012648) q[0];
ry(-0.6939952819254938) q[1];
rz(-1.6512465235136162) q[1];
ry(-1.1953859097863322) q[2];
rz(-1.2567170319013252) q[2];
ry(-2.5170060683813036) q[3];
rz(2.3761889670075904) q[3];
ry(-0.1648478234664621) q[4];
rz(2.440023331377536) q[4];
ry(-2.1959695719586985) q[5];
rz(1.737382430978371) q[5];
ry(-2.8287089717711287) q[6];
rz(-0.7569578733806941) q[6];
ry(1.8710511461089292) q[7];
rz(2.6635850382999835) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.717052507304472) q[0];
rz(-2.311219719926402) q[0];
ry(-3.057037260390901) q[1];
rz(-0.141942023422982) q[1];
ry(-1.5021145011804136) q[2];
rz(2.577084474827788) q[2];
ry(0.49060878762992305) q[3];
rz(-2.6362436498700434) q[3];
ry(-2.8423441208949374) q[4];
rz(0.13317693201121564) q[4];
ry(-1.5984019990937628) q[5];
rz(3.0871686900351) q[5];
ry(-1.4476404197401234) q[6];
rz(-1.9264377130192543) q[6];
ry(-2.2118149025920903) q[7];
rz(2.634073753895276) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.4548899513802727) q[0];
rz(-2.79315197778579) q[0];
ry(-1.6373600641076145) q[1];
rz(-1.0855372319479688) q[1];
ry(0.33188441619182274) q[2];
rz(-1.2160996249621658) q[2];
ry(0.5903112460785946) q[3];
rz(3.1116719915345334) q[3];
ry(3.03914438264177) q[4];
rz(0.44339778028490684) q[4];
ry(0.20026792655130615) q[5];
rz(1.7301336389378454) q[5];
ry(-0.17026226005725914) q[6];
rz(-0.9230341867315754) q[6];
ry(1.845541485360937) q[7];
rz(-0.7693120111650016) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.9281230841659482) q[0];
rz(1.8200811404979425) q[0];
ry(2.123351892448954) q[1];
rz(-1.293153343032932) q[1];
ry(-2.4791428676614453) q[2];
rz(-2.5087653758633537) q[2];
ry(0.22704899842801338) q[3];
rz(-0.545765943449804) q[3];
ry(-0.21289778967607284) q[4];
rz(0.6486611174216383) q[4];
ry(-2.3457631982564506) q[5];
rz(-2.801754578458686) q[5];
ry(-2.329435875731064) q[6];
rz(2.941689063355007) q[6];
ry(0.13019784994292483) q[7];
rz(0.8457530702266997) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.0443974587592626) q[0];
rz(-0.09062730403278439) q[0];
ry(0.04777170138597185) q[1];
rz(0.6327711781340711) q[1];
ry(3.1401213050115424) q[2];
rz(-2.0661035186503725) q[2];
ry(3.092886672178565) q[3];
rz(0.8902642014475575) q[3];
ry(-2.932076976024049) q[4];
rz(1.051192757545307) q[4];
ry(-0.7145207540187645) q[5];
rz(0.24074948606842086) q[5];
ry(-2.7170418113399966) q[6];
rz(3.0549167526136745) q[6];
ry(1.1397441442238954) q[7];
rz(1.0334413268735965) q[7];