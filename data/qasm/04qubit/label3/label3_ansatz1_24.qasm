OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.5696629654396412) q[0];
rz(1.2054331328263828) q[0];
ry(2.0090695964208383) q[1];
rz(-2.2895546950254464) q[1];
ry(-1.8208353099876815) q[2];
rz(0.0324006690087284) q[2];
ry(-2.4863189399792116) q[3];
rz(2.268880019844585) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7952208301030845) q[0];
rz(1.8305793824939587) q[0];
ry(0.5982203017276259) q[1];
rz(0.789995045213109) q[1];
ry(0.7578792951416097) q[2];
rz(-0.674309203773519) q[2];
ry(-2.13773441214909) q[3];
rz(-1.4179491797057127) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6111233338147901) q[0];
rz(-1.8474202521159988) q[0];
ry(0.7072567947457469) q[1];
rz(-2.641900963186811) q[1];
ry(-0.10835339804791744) q[2];
rz(-2.788333932228745) q[2];
ry(0.18724581276547436) q[3];
rz(-1.904847810116728) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.982624426466139) q[0];
rz(0.9709629552687835) q[0];
ry(-2.3579548525680436) q[1];
rz(2.9452382441232836) q[1];
ry(2.731041008038745) q[2];
rz(2.756453114749892) q[2];
ry(-1.702962117340676) q[3];
rz(-0.9490495415952144) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7583387294277344) q[0];
rz(-1.3252304632876764) q[0];
ry(2.4021407359945144) q[1];
rz(2.8128884271319485) q[1];
ry(0.32152682204663474) q[2];
rz(-3.0923149344284875) q[2];
ry(1.3366682062244841) q[3];
rz(-0.14348208128856077) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.1198984049229423) q[0];
rz(2.6523700719684817) q[0];
ry(-1.5960752275959642) q[1];
rz(2.063762667015064) q[1];
ry(-0.6360485547458028) q[2];
rz(0.5749411564588724) q[2];
ry(2.305999151254397) q[3];
rz(1.6749510116975186) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.60808509945877) q[0];
rz(-0.6862744575678993) q[0];
ry(-1.6940430268375268) q[1];
rz(2.1394044636718386) q[1];
ry(-2.4800021187942507) q[2];
rz(2.2477816910650814) q[2];
ry(-0.26858184986239236) q[3];
rz(2.432266457061918) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4879988814716794) q[0];
rz(1.4231076592972545) q[0];
ry(-1.7134990302719342) q[1];
rz(0.7981112349431142) q[1];
ry(2.968008726828064) q[2];
rz(-0.8973984260038191) q[2];
ry(1.55840352142629) q[3];
rz(-2.9765803495451637) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.7746266353297098) q[0];
rz(-2.2438467546953555) q[0];
ry(1.63341876622058) q[1];
rz(0.012794978282057765) q[1];
ry(0.3180915682761513) q[2];
rz(2.542127213238872) q[2];
ry(-3.0743850951922065) q[3];
rz(0.7068390296442207) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.1224100914852784) q[0];
rz(0.7114672996143705) q[0];
ry(1.3720075915145031) q[1];
rz(-2.365013083712639) q[1];
ry(-1.9747091426610082) q[2];
rz(-0.3861009910117206) q[2];
ry(1.4252435178741296) q[3];
rz(-1.791387763366811) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6064196728281943) q[0];
rz(3.04841118924999) q[0];
ry(-2.337737520686033) q[1];
rz(-2.433840268888703) q[1];
ry(2.0297856875077436) q[2];
rz(2.313074731725617) q[2];
ry(1.4963291255049782) q[3];
rz(-1.0087229103625033) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6140885144712719) q[0];
rz(1.6140461607000183) q[0];
ry(0.2350338463950603) q[1];
rz(-0.5657168425472862) q[1];
ry(-1.0743544499616366) q[2];
rz(-2.6105984553055457) q[2];
ry(0.5423351207597914) q[3];
rz(1.687251864863595) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0153658182601895) q[0];
rz(1.8639716212368844) q[0];
ry(1.9065572703688929) q[1];
rz(0.881142178827818) q[1];
ry(-0.3761801083704883) q[2];
rz(1.8165889206316503) q[2];
ry(-0.8811076719876487) q[3];
rz(-0.7811320275399103) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.1826239211753675) q[0];
rz(1.7381755341053384) q[0];
ry(-2.9558470148651392) q[1];
rz(-1.1092302066035746) q[1];
ry(1.548709005384631) q[2];
rz(-2.7044929173918186) q[2];
ry(-1.372397943954705) q[3];
rz(-1.1826319495169166) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.8405569891994519) q[0];
rz(-2.1678485325355186) q[0];
ry(-0.7204198290836397) q[1];
rz(-1.2127108513832132) q[1];
ry(1.0815103582154004) q[2];
rz(-2.5690341446390352) q[2];
ry(-2.433910005537377) q[3];
rz(2.7373293181405027) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.58835773562915) q[0];
rz(2.742443724675702) q[0];
ry(1.3082457383354678) q[1];
rz(2.615092042777362) q[1];
ry(0.7283609990911266) q[2];
rz(-2.048351700323436) q[2];
ry(-0.5055279836733222) q[3];
rz(-1.7239595231772133) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7971454179836741) q[0];
rz(-2.3332002672492504) q[0];
ry(1.0012730106526844) q[1];
rz(1.073973151260609) q[1];
ry(-2.4006988372158857) q[2];
rz(-2.5345949658014004) q[2];
ry(-1.1103603972450902) q[3];
rz(1.2875471477363043) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.3152080987666177) q[0];
rz(-2.7941896429362583) q[0];
ry(1.7070675581852228) q[1];
rz(-1.6613949871418416) q[1];
ry(2.3635436274977515) q[2];
rz(1.6617344875386668) q[2];
ry(-1.4565961230142452) q[3];
rz(0.3894001210741602) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.1362663555246733) q[0];
rz(-0.4916115111760986) q[0];
ry(0.6723533886402926) q[1];
rz(-1.3527128380699311) q[1];
ry(-2.1765697299965288) q[2];
rz(2.8716350047206314) q[2];
ry(-3.03523253710232) q[3];
rz(-1.6010424128367564) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.0789455477301484) q[0];
rz(2.1107287138582698) q[0];
ry(1.1932978122176188) q[1];
rz(-2.158577222745185) q[1];
ry(2.7925078041423337) q[2];
rz(-3.0525956425480274) q[2];
ry(0.9931369293722536) q[3];
rz(-1.3060256039435973) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.3969898714299642) q[0];
rz(0.3783840299942947) q[0];
ry(0.5044388252817832) q[1];
rz(0.7278192378607117) q[1];
ry(-0.19463185565836771) q[2];
rz(-0.15775365810064496) q[2];
ry(0.8783751224178271) q[3];
rz(2.8464674087653568) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.8530199804242723) q[0];
rz(1.694541099832442) q[0];
ry(-1.4488205792389908) q[1];
rz(0.40861726572440027) q[1];
ry(1.9492686679251487) q[2];
rz(0.8958125698733412) q[2];
ry(3.109229564804715) q[3];
rz(2.8386727649625745) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4177800186900895) q[0];
rz(1.4780995931750538) q[0];
ry(-2.5524425096804015) q[1];
rz(-0.0008824747567072049) q[1];
ry(1.9861493546329883) q[2];
rz(1.6471653458503592) q[2];
ry(-1.3461783521306823) q[3];
rz(-2.2193487608391758) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2190802631766402) q[0];
rz(-0.13329481770672627) q[0];
ry(0.37894846933888204) q[1];
rz(-2.735111487759154) q[1];
ry(0.48273123309945476) q[2];
rz(-2.5325524057861553) q[2];
ry(0.7561833585997517) q[3];
rz(0.4973311613675707) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.2008199109225943) q[0];
rz(0.6160482977251398) q[0];
ry(2.443600889808357) q[1];
rz(-1.282505931774826) q[1];
ry(0.06262710511905577) q[2];
rz(2.247768204540805) q[2];
ry(-1.1549987784091353) q[3];
rz(-1.0406266442613448) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8022067515609317) q[0];
rz(-2.7899829149317608) q[0];
ry(3.0187220393991434) q[1];
rz(-0.08082336560522142) q[1];
ry(-1.8258682995860251) q[2];
rz(-2.8627513156988025) q[2];
ry(2.5094204254604446) q[3];
rz(2.5783539735315912) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3876639495818344) q[0];
rz(0.3458717202525285) q[0];
ry(-1.2969190681876164) q[1];
rz(-0.7622313285749005) q[1];
ry(1.591826405823248) q[2];
rz(-0.5026686987374074) q[2];
ry(0.2874630470524364) q[3];
rz(2.3948997658134745) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0066617572903027) q[0];
rz(1.1366573753475242) q[0];
ry(-2.047984165480773) q[1];
rz(-2.988831043930619) q[1];
ry(-0.22768101992493062) q[2];
rz(-2.991210536017165) q[2];
ry(-0.4320507683944653) q[3];
rz(0.4857783434979304) q[3];