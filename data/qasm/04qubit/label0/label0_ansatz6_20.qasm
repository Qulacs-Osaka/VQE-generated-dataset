OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.140971004833065) q[0];
ry(2.483020995882831) q[1];
cx q[0],q[1];
ry(-1.2154757664132536) q[0];
ry(-0.5898692615068146) q[1];
cx q[0],q[1];
ry(-0.4756161004378688) q[1];
ry(-2.334315020529135) q[2];
cx q[1],q[2];
ry(0.27270949858561955) q[1];
ry(-3.021608320518354) q[2];
cx q[1],q[2];
ry(2.7931762986491377) q[2];
ry(1.7105073428616322) q[3];
cx q[2],q[3];
ry(-1.0164177088545654) q[2];
ry(-2.701014266815067) q[3];
cx q[2],q[3];
ry(1.9022938689071607) q[0];
ry(2.7293563957855542) q[1];
cx q[0],q[1];
ry(-2.3372573443232394) q[0];
ry(-0.5795414423975622) q[1];
cx q[0],q[1];
ry(-1.1916286292837635) q[1];
ry(0.7692499873143808) q[2];
cx q[1],q[2];
ry(1.1707747531713935) q[1];
ry(0.676689683391692) q[2];
cx q[1],q[2];
ry(0.3176346990826333) q[2];
ry(1.5762557395966255) q[3];
cx q[2],q[3];
ry(-2.4972365319334067) q[2];
ry(-2.0605479029894047) q[3];
cx q[2],q[3];
ry(1.8910587763192765) q[0];
ry(-2.7734940877519674) q[1];
cx q[0],q[1];
ry(0.4589630022825357) q[0];
ry(0.4433189355428792) q[1];
cx q[0],q[1];
ry(1.173236585345993) q[1];
ry(-1.5807582457322589) q[2];
cx q[1],q[2];
ry(2.808560981906205) q[1];
ry(-2.5002604509013544) q[2];
cx q[1],q[2];
ry(0.5568389262315572) q[2];
ry(-2.737002323740008) q[3];
cx q[2],q[3];
ry(0.4729310891750264) q[2];
ry(-1.5472233658465662) q[3];
cx q[2],q[3];
ry(-2.7441727761008003) q[0];
ry(0.45557215174777) q[1];
cx q[0],q[1];
ry(1.5181043663502427) q[0];
ry(-3.085138163562853) q[1];
cx q[0],q[1];
ry(0.23813761596368632) q[1];
ry(-0.377467692923493) q[2];
cx q[1],q[2];
ry(2.5974627648603668) q[1];
ry(1.9445413324682335) q[2];
cx q[1],q[2];
ry(1.2011004009138027) q[2];
ry(0.06837995320455724) q[3];
cx q[2],q[3];
ry(2.4896721293998714) q[2];
ry(1.9193006516007183) q[3];
cx q[2],q[3];
ry(0.5924830339987412) q[0];
ry(-0.734304562208191) q[1];
cx q[0],q[1];
ry(-0.9236691085191788) q[0];
ry(-0.7898959356765172) q[1];
cx q[0],q[1];
ry(2.031789724801239) q[1];
ry(-2.9889326008535244) q[2];
cx q[1],q[2];
ry(2.333323604346618) q[1];
ry(-2.4906081961390014) q[2];
cx q[1],q[2];
ry(0.759421468268845) q[2];
ry(-0.4606159632598818) q[3];
cx q[2],q[3];
ry(-1.3291905127269872) q[2];
ry(1.6180072024185865) q[3];
cx q[2],q[3];
ry(-1.9238476249705458) q[0];
ry(0.9322941883263596) q[1];
cx q[0],q[1];
ry(2.994308658374279) q[0];
ry(-0.7081383561534136) q[1];
cx q[0],q[1];
ry(-0.1367477235279108) q[1];
ry(2.0839272968366633) q[2];
cx q[1],q[2];
ry(-1.8820848438245072) q[1];
ry(-0.854664592008568) q[2];
cx q[1],q[2];
ry(2.6234598028653755) q[2];
ry(1.900812514027616) q[3];
cx q[2],q[3];
ry(1.845816636894915) q[2];
ry(-2.689103267011555) q[3];
cx q[2],q[3];
ry(-1.8503621051928576) q[0];
ry(-0.526871134924426) q[1];
cx q[0],q[1];
ry(-2.5383375619577735) q[0];
ry(-2.0410611027057053) q[1];
cx q[0],q[1];
ry(-1.9026428005726128) q[1];
ry(0.5274940012326792) q[2];
cx q[1],q[2];
ry(0.7235223789907543) q[1];
ry(1.1785137356456186) q[2];
cx q[1],q[2];
ry(1.5623558455809967) q[2];
ry(1.3818956756068355) q[3];
cx q[2],q[3];
ry(-2.262065837268352) q[2];
ry(1.1872068143062509) q[3];
cx q[2],q[3];
ry(-1.9371902859066745) q[0];
ry(0.6548214905456133) q[1];
cx q[0],q[1];
ry(0.8411437489841818) q[0];
ry(2.097417239368024) q[1];
cx q[0],q[1];
ry(-3.069172101342878) q[1];
ry(1.5116996488446124) q[2];
cx q[1],q[2];
ry(-1.8401676344395648) q[1];
ry(-2.132499237488175) q[2];
cx q[1],q[2];
ry(1.9844303309491833) q[2];
ry(0.5456295992539639) q[3];
cx q[2],q[3];
ry(1.4531927688369988) q[2];
ry(-2.453067043193726) q[3];
cx q[2],q[3];
ry(-1.6796728859392944) q[0];
ry(-1.000604849769665) q[1];
cx q[0],q[1];
ry(-2.9787861456285385) q[0];
ry(-1.1091956196666095) q[1];
cx q[0],q[1];
ry(-1.0088721486610481) q[1];
ry(2.5769609274529226) q[2];
cx q[1],q[2];
ry(1.8128889989937296) q[1];
ry(2.3523999940925098) q[2];
cx q[1],q[2];
ry(-2.748671412615389) q[2];
ry(1.9890792367347099) q[3];
cx q[2],q[3];
ry(-0.7606723709179449) q[2];
ry(-1.0682267550946705) q[3];
cx q[2],q[3];
ry(-0.4425212032898944) q[0];
ry(-2.881403186208051) q[1];
cx q[0],q[1];
ry(-2.1071509905416725) q[0];
ry(2.453784692936978) q[1];
cx q[0],q[1];
ry(2.6196828012976447) q[1];
ry(-0.3145068273229201) q[2];
cx q[1],q[2];
ry(2.69718437001401) q[1];
ry(-2.7882478227676137) q[2];
cx q[1],q[2];
ry(0.09648340092278841) q[2];
ry(2.055255433057859) q[3];
cx q[2],q[3];
ry(1.4648731797078425) q[2];
ry(-0.967099968507469) q[3];
cx q[2],q[3];
ry(-2.207734906989385) q[0];
ry(1.9336168114032368) q[1];
cx q[0],q[1];
ry(-2.6591300009145193) q[0];
ry(-1.6342472887261819) q[1];
cx q[0],q[1];
ry(-1.2646869791567266) q[1];
ry(1.2584406388534175) q[2];
cx q[1],q[2];
ry(-1.5714325211346007) q[1];
ry(-0.9359941320406042) q[2];
cx q[1],q[2];
ry(0.6208358780885833) q[2];
ry(2.717199289872098) q[3];
cx q[2],q[3];
ry(1.0116762657832883) q[2];
ry(-2.0457579735868734) q[3];
cx q[2],q[3];
ry(2.5525420800932306) q[0];
ry(-1.9510851620638083) q[1];
cx q[0],q[1];
ry(1.6522858708872243) q[0];
ry(0.7565191201367697) q[1];
cx q[0],q[1];
ry(1.5630845700093472) q[1];
ry(0.6870223048265967) q[2];
cx q[1],q[2];
ry(-2.8331106456762605) q[1];
ry(2.4979435273779944) q[2];
cx q[1],q[2];
ry(-1.2995854509459408) q[2];
ry(1.6815417181335635) q[3];
cx q[2],q[3];
ry(0.4367618861880448) q[2];
ry(-0.6330018534881496) q[3];
cx q[2],q[3];
ry(-2.679292524112448) q[0];
ry(1.8053661095359275) q[1];
cx q[0],q[1];
ry(-1.5077310826086148) q[0];
ry(2.0534574669265675) q[1];
cx q[0],q[1];
ry(-1.2921739081625496) q[1];
ry(-2.6421647021428054) q[2];
cx q[1],q[2];
ry(-1.048827279430256) q[1];
ry(-0.986348551567902) q[2];
cx q[1],q[2];
ry(-0.4838644507968916) q[2];
ry(-0.9890485914432625) q[3];
cx q[2],q[3];
ry(-1.6852785494936136) q[2];
ry(1.63670794868046) q[3];
cx q[2],q[3];
ry(-1.2056353655869598) q[0];
ry(-0.17584161719488112) q[1];
cx q[0],q[1];
ry(-2.627573175170261) q[0];
ry(0.09824975488560318) q[1];
cx q[0],q[1];
ry(2.2551992134691443) q[1];
ry(0.6223720534619837) q[2];
cx q[1],q[2];
ry(0.44667758042297173) q[1];
ry(2.660107963526342) q[2];
cx q[1],q[2];
ry(2.3362902372517413) q[2];
ry(-2.345278140624611) q[3];
cx q[2],q[3];
ry(1.109169864602987) q[2];
ry(2.4494357616553275) q[3];
cx q[2],q[3];
ry(0.1918739941483709) q[0];
ry(-2.716009261111969) q[1];
cx q[0],q[1];
ry(1.2770811977538115) q[0];
ry(-1.968971893240991) q[1];
cx q[0],q[1];
ry(0.11635025721158487) q[1];
ry(1.909756671304424) q[2];
cx q[1],q[2];
ry(-2.2200615055986583) q[1];
ry(1.9585486351870438) q[2];
cx q[1],q[2];
ry(0.19727728569923336) q[2];
ry(-2.8677592929843563) q[3];
cx q[2],q[3];
ry(-0.7067351567796019) q[2];
ry(1.148993078867953) q[3];
cx q[2],q[3];
ry(1.3572417084515365) q[0];
ry(0.5562019799034824) q[1];
cx q[0],q[1];
ry(2.3432380749231796) q[0];
ry(-1.597669441000174) q[1];
cx q[0],q[1];
ry(2.6059557282733565) q[1];
ry(0.691261104600843) q[2];
cx q[1],q[2];
ry(-1.9244923579675903) q[1];
ry(-1.3570975762021364) q[2];
cx q[1],q[2];
ry(-0.3270956018166302) q[2];
ry(1.088250157102924) q[3];
cx q[2],q[3];
ry(-1.5219945705535425) q[2];
ry(0.2487145851126681) q[3];
cx q[2],q[3];
ry(-1.7880101104569317) q[0];
ry(2.6371728049852394) q[1];
cx q[0],q[1];
ry(0.7193794666039558) q[0];
ry(-2.0869111393391897) q[1];
cx q[0],q[1];
ry(-2.526164718252496) q[1];
ry(-1.428430464199586) q[2];
cx q[1],q[2];
ry(-1.353251639096124) q[1];
ry(0.7965763876183907) q[2];
cx q[1],q[2];
ry(0.22549084532343144) q[2];
ry(2.889088532198908) q[3];
cx q[2],q[3];
ry(-1.993576015923516) q[2];
ry(-2.0594556678146465) q[3];
cx q[2],q[3];
ry(-1.9220882366956427) q[0];
ry(3.0046804210612272) q[1];
cx q[0],q[1];
ry(1.9936979375874975) q[0];
ry(2.615791919111842) q[1];
cx q[0],q[1];
ry(0.9487308821685725) q[1];
ry(-2.9853284319735103) q[2];
cx q[1],q[2];
ry(0.48208627091122036) q[1];
ry(1.2822431856773324) q[2];
cx q[1],q[2];
ry(-1.9537828334490603) q[2];
ry(1.140749126726659) q[3];
cx q[2],q[3];
ry(-0.9902993111799949) q[2];
ry(0.08160152193457987) q[3];
cx q[2],q[3];
ry(1.6023145491080335) q[0];
ry(0.6021273700258414) q[1];
cx q[0],q[1];
ry(-1.9111956526804548) q[0];
ry(0.42861574117267853) q[1];
cx q[0],q[1];
ry(-3.0266708348928977) q[1];
ry(2.9415670445306823) q[2];
cx q[1],q[2];
ry(-2.55173474144935) q[1];
ry(-2.4928988746882257) q[2];
cx q[1],q[2];
ry(-0.4718634615995782) q[2];
ry(-0.0699830226529943) q[3];
cx q[2],q[3];
ry(1.8816250840930986) q[2];
ry(1.962001837209864) q[3];
cx q[2],q[3];
ry(-0.12853427971075781) q[0];
ry(2.890337189500771) q[1];
cx q[0],q[1];
ry(-0.5649887074464743) q[0];
ry(-1.4028043539347843) q[1];
cx q[0],q[1];
ry(0.34167470008852724) q[1];
ry(-2.1472531400712116) q[2];
cx q[1],q[2];
ry(-0.6578126075566448) q[1];
ry(-2.8554227781879327) q[2];
cx q[1],q[2];
ry(1.7477816189525273) q[2];
ry(1.4579751019430223) q[3];
cx q[2],q[3];
ry(1.8169077948266557) q[2];
ry(-0.19098578908724986) q[3];
cx q[2],q[3];
ry(3.1103315880507116) q[0];
ry(0.28609888812311557) q[1];
cx q[0],q[1];
ry(0.45331500408026965) q[0];
ry(0.877781445379882) q[1];
cx q[0],q[1];
ry(2.362082844799335) q[1];
ry(0.3505179925473412) q[2];
cx q[1],q[2];
ry(-0.22324892162656618) q[1];
ry(-0.9662319806950109) q[2];
cx q[1],q[2];
ry(-0.9266802955500104) q[2];
ry(2.3335610809428884) q[3];
cx q[2],q[3];
ry(-1.8159155189556757) q[2];
ry(-1.8769600073252808) q[3];
cx q[2],q[3];
ry(-1.2213967848063643) q[0];
ry(-0.3876630594058925) q[1];
cx q[0],q[1];
ry(-2.4647538646246008) q[0];
ry(3.1129469459622996) q[1];
cx q[0],q[1];
ry(2.2569125907354266) q[1];
ry(-0.2748220992434223) q[2];
cx q[1],q[2];
ry(1.548926496829053) q[1];
ry(0.6190423273708838) q[2];
cx q[1],q[2];
ry(-0.6284485408826761) q[2];
ry(2.4488137634170637) q[3];
cx q[2],q[3];
ry(1.4601119940271918) q[2];
ry(-2.523682343652088) q[3];
cx q[2],q[3];
ry(-1.2444555914210813) q[0];
ry(2.0106082517635198) q[1];
cx q[0],q[1];
ry(-2.266015666681554) q[0];
ry(-1.0045732095619968) q[1];
cx q[0],q[1];
ry(0.37990489350609397) q[1];
ry(1.3285548716592293) q[2];
cx q[1],q[2];
ry(0.9141664667849867) q[1];
ry(-0.6311955960845697) q[2];
cx q[1],q[2];
ry(0.3723015795383144) q[2];
ry(1.1109071083792772) q[3];
cx q[2],q[3];
ry(-2.051403203470446) q[2];
ry(-2.9180380656456975) q[3];
cx q[2],q[3];
ry(-1.6077620563934987) q[0];
ry(-2.644708859274914) q[1];
ry(-1.5134746431172905) q[2];
ry(3.1058641887306218) q[3];