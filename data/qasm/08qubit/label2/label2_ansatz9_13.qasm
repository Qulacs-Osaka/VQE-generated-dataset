OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.016987004402895) q[0];
ry(1.7625323057853026) q[1];
cx q[0],q[1];
ry(0.20798798476416833) q[0];
ry(0.6365741823156625) q[1];
cx q[0],q[1];
ry(0.7804199660697835) q[2];
ry(0.6821585713616921) q[3];
cx q[2],q[3];
ry(-0.45179333275689965) q[2];
ry(-1.1775273783691595) q[3];
cx q[2],q[3];
ry(-0.8872694662740962) q[4];
ry(1.6854167151724726) q[5];
cx q[4],q[5];
ry(0.7646364608570498) q[4];
ry(-3.048138751119453) q[5];
cx q[4],q[5];
ry(1.952746227476198) q[6];
ry(2.4622314650534776) q[7];
cx q[6],q[7];
ry(1.404915101573897) q[6];
ry(-1.803927010201634) q[7];
cx q[6],q[7];
ry(-1.8649230427105175) q[0];
ry(-1.0354201033892316) q[2];
cx q[0],q[2];
ry(-2.24277187161821) q[0];
ry(-1.2540232872688255) q[2];
cx q[0],q[2];
ry(3.1365673054926826) q[2];
ry(-0.15332062224667364) q[4];
cx q[2],q[4];
ry(-1.1596254324636028) q[2];
ry(-1.5194719034288335) q[4];
cx q[2],q[4];
ry(0.25031566138971506) q[4];
ry(1.9874554773194983) q[6];
cx q[4],q[6];
ry(0.0946050197754972) q[4];
ry(2.4977532410227226) q[6];
cx q[4],q[6];
ry(-2.763440522715556) q[1];
ry(-0.7834627314010989) q[3];
cx q[1],q[3];
ry(-2.568222462448265) q[1];
ry(2.413502647649747) q[3];
cx q[1],q[3];
ry(-1.1228626343155244) q[3];
ry(-0.6379961155545331) q[5];
cx q[3],q[5];
ry(1.7863421317861155) q[3];
ry(1.5582581521266778) q[5];
cx q[3],q[5];
ry(2.713031508832114) q[5];
ry(-1.7651940863703715) q[7];
cx q[5],q[7];
ry(-1.6321991323420897) q[5];
ry(1.3559256573603669) q[7];
cx q[5],q[7];
ry(2.3928863585605007) q[0];
ry(1.4519429060579143) q[3];
cx q[0],q[3];
ry(-0.8916598219618725) q[0];
ry(0.5189230405675145) q[3];
cx q[0],q[3];
ry(1.9565685975748206) q[1];
ry(-0.04927312445445666) q[2];
cx q[1],q[2];
ry(-0.7337574558138256) q[1];
ry(1.642390636391917) q[2];
cx q[1],q[2];
ry(2.8482340430553967) q[2];
ry(2.798221102962468) q[5];
cx q[2],q[5];
ry(1.7005649637442621) q[2];
ry(1.6289302168139952) q[5];
cx q[2],q[5];
ry(1.3647744314866097) q[3];
ry(0.5896994087239945) q[4];
cx q[3],q[4];
ry(1.4719716384169619) q[3];
ry(1.975964586992255) q[4];
cx q[3],q[4];
ry(1.1383384855462708) q[4];
ry(1.6911685952814555) q[7];
cx q[4],q[7];
ry(-1.306830048908716) q[4];
ry(-1.2437440513398916) q[7];
cx q[4],q[7];
ry(1.643232221979468) q[5];
ry(3.027264420046881) q[6];
cx q[5],q[6];
ry(-1.5257849059524413) q[5];
ry(3.1415000782286517) q[6];
cx q[5],q[6];
ry(-2.7331302499804813) q[0];
ry(-2.8200387054638356) q[1];
cx q[0],q[1];
ry(-1.9968666436816853) q[0];
ry(1.6297023586004213) q[1];
cx q[0],q[1];
ry(-1.3729743126358882) q[2];
ry(-0.8885379230337661) q[3];
cx q[2],q[3];
ry(-1.615470373335345) q[2];
ry(0.9542132506875243) q[3];
cx q[2],q[3];
ry(0.3952551861155057) q[4];
ry(-2.80910487060621) q[5];
cx q[4],q[5];
ry(1.4988088171445297) q[4];
ry(1.22897695459982) q[5];
cx q[4],q[5];
ry(2.695597515214731) q[6];
ry(-0.9887115991929329) q[7];
cx q[6],q[7];
ry(0.21095254065328553) q[6];
ry(2.068925421150014) q[7];
cx q[6],q[7];
ry(-1.6922269524554794) q[0];
ry(0.9329678584458723) q[2];
cx q[0],q[2];
ry(1.08701932092493) q[0];
ry(0.8003089712001882) q[2];
cx q[0],q[2];
ry(1.7465734256615721) q[2];
ry(-1.0711388516158344) q[4];
cx q[2],q[4];
ry(1.296607736711481) q[2];
ry(-2.0745803503542177) q[4];
cx q[2],q[4];
ry(-3.0568661199344036) q[4];
ry(2.5015948408671935) q[6];
cx q[4],q[6];
ry(0.039365340293215034) q[4];
ry(0.8652015857149199) q[6];
cx q[4],q[6];
ry(2.4149954002749348) q[1];
ry(-1.6483515376924094) q[3];
cx q[1],q[3];
ry(1.3220584443152248) q[1];
ry(0.9656995950845539) q[3];
cx q[1],q[3];
ry(2.337138390776684) q[3];
ry(-0.9478506199486016) q[5];
cx q[3],q[5];
ry(2.6680944212047386) q[3];
ry(-2.4321540219529774) q[5];
cx q[3],q[5];
ry(-2.1535528148895873) q[5];
ry(-3.0893965063780042) q[7];
cx q[5],q[7];
ry(0.7266778372347877) q[5];
ry(-1.8289878588786037) q[7];
cx q[5],q[7];
ry(1.7925574464963863) q[0];
ry(-1.6410882267584395) q[3];
cx q[0],q[3];
ry(-0.38441677382781003) q[0];
ry(1.5697736277184813) q[3];
cx q[0],q[3];
ry(2.2548746048041073) q[1];
ry(-1.3315501649285109) q[2];
cx q[1],q[2];
ry(2.4692017189399684) q[1];
ry(-1.3552660032508355) q[2];
cx q[1],q[2];
ry(-0.0799064261060719) q[2];
ry(0.10556458155293931) q[5];
cx q[2],q[5];
ry(2.5375131549133103) q[2];
ry(-2.3479056031841816) q[5];
cx q[2],q[5];
ry(-2.1132676632994505) q[3];
ry(-1.0528042627825456) q[4];
cx q[3],q[4];
ry(2.3726245915186115) q[3];
ry(1.7019233416683397) q[4];
cx q[3],q[4];
ry(-0.01827739481737517) q[4];
ry(0.8077605066196405) q[7];
cx q[4],q[7];
ry(-2.5347494448381958) q[4];
ry(-0.2443162561501242) q[7];
cx q[4],q[7];
ry(1.4774938190532667) q[5];
ry(-2.563352663151263) q[6];
cx q[5],q[6];
ry(-1.6088123242018189) q[5];
ry(1.9383568881944493) q[6];
cx q[5],q[6];
ry(-0.7435102675805948) q[0];
ry(-2.7068552493667064) q[1];
cx q[0],q[1];
ry(-0.11940861941978831) q[0];
ry(-1.6959285685524899) q[1];
cx q[0],q[1];
ry(-1.61837351412201) q[2];
ry(-1.1535460011980534) q[3];
cx q[2],q[3];
ry(3.091195528884095) q[2];
ry(-1.263134903611812) q[3];
cx q[2],q[3];
ry(-1.4527929138422049) q[4];
ry(0.549317473217478) q[5];
cx q[4],q[5];
ry(-2.0448884197854795) q[4];
ry(1.7111185862174751) q[5];
cx q[4],q[5];
ry(-2.7004543605723663) q[6];
ry(1.8885529500904215) q[7];
cx q[6],q[7];
ry(2.071943356685203) q[6];
ry(-2.8657840333034605) q[7];
cx q[6],q[7];
ry(0.12494909588351055) q[0];
ry(-0.7666831897442954) q[2];
cx q[0],q[2];
ry(-1.32622298500503) q[0];
ry(2.849109976009842) q[2];
cx q[0],q[2];
ry(-1.9328519442010188) q[2];
ry(0.750615611745757) q[4];
cx q[2],q[4];
ry(3.03498264799796) q[2];
ry(-2.866267524387865) q[4];
cx q[2],q[4];
ry(0.8365127562048181) q[4];
ry(-2.7206497253279336) q[6];
cx q[4],q[6];
ry(1.767126056432371) q[4];
ry(0.05023319455778452) q[6];
cx q[4],q[6];
ry(-3.0340892656898597) q[1];
ry(-1.1170586063797492) q[3];
cx q[1],q[3];
ry(-1.0032251499606395) q[1];
ry(-1.3807122710542472) q[3];
cx q[1],q[3];
ry(0.3407567890214933) q[3];
ry(-1.2972923791255853) q[5];
cx q[3],q[5];
ry(0.6323303965013247) q[3];
ry(-2.960708813091436) q[5];
cx q[3],q[5];
ry(-1.1863286997307414) q[5];
ry(0.3547369791528414) q[7];
cx q[5],q[7];
ry(-2.1344982837473196) q[5];
ry(1.4852714981941602) q[7];
cx q[5],q[7];
ry(-1.5071901751698076) q[0];
ry(2.801769618394772) q[3];
cx q[0],q[3];
ry(2.6282159762274873) q[0];
ry(2.881660567554957) q[3];
cx q[0],q[3];
ry(1.798141735159657) q[1];
ry(-2.2720564132089454) q[2];
cx q[1],q[2];
ry(-1.2008128575782884) q[1];
ry(-1.8573615194651527) q[2];
cx q[1],q[2];
ry(-2.376124496608503) q[2];
ry(2.3323153847024067) q[5];
cx q[2],q[5];
ry(-0.6288671117857056) q[2];
ry(-3.0163562437803) q[5];
cx q[2],q[5];
ry(-0.8892352187899704) q[3];
ry(-1.399727699312872) q[4];
cx q[3],q[4];
ry(1.2893674107064985) q[3];
ry(1.2365187577924237) q[4];
cx q[3],q[4];
ry(-1.8752309197813641) q[4];
ry(-1.7614587467371798) q[7];
cx q[4],q[7];
ry(-2.783830803009736) q[4];
ry(-0.1400788982945036) q[7];
cx q[4],q[7];
ry(0.047209734557625546) q[5];
ry(2.938385200287771) q[6];
cx q[5],q[6];
ry(2.7324674565537332) q[5];
ry(0.6057446233239974) q[6];
cx q[5],q[6];
ry(-1.6866407524049079) q[0];
ry(-2.892117415523394) q[1];
cx q[0],q[1];
ry(-2.06339931093323) q[0];
ry(1.4733222638357848) q[1];
cx q[0],q[1];
ry(2.1685514575984266) q[2];
ry(3.016943617351414) q[3];
cx q[2],q[3];
ry(0.0749492151874529) q[2];
ry(-1.2257835650436784) q[3];
cx q[2],q[3];
ry(2.3976188719269222) q[4];
ry(2.3754460705688487) q[5];
cx q[4],q[5];
ry(-1.4362336732162735) q[4];
ry(-2.0694749369791694) q[5];
cx q[4],q[5];
ry(2.0260450647521964) q[6];
ry(2.072885215990998) q[7];
cx q[6],q[7];
ry(-3.0354661240995) q[6];
ry(-1.3372674894339704) q[7];
cx q[6],q[7];
ry(1.1391091496831578) q[0];
ry(-2.846238013106353) q[2];
cx q[0],q[2];
ry(1.7017217180682396) q[0];
ry(-1.8056413759757544) q[2];
cx q[0],q[2];
ry(0.16212247318505657) q[2];
ry(2.5552281073829435) q[4];
cx q[2],q[4];
ry(-1.5676795569588549) q[2];
ry(-0.0015489771519565374) q[4];
cx q[2],q[4];
ry(1.8675716545254704) q[4];
ry(0.4137025256800552) q[6];
cx q[4],q[6];
ry(-2.095850352833181) q[4];
ry(1.4120758516756826) q[6];
cx q[4],q[6];
ry(0.5619021222079219) q[1];
ry(-1.869575378556365) q[3];
cx q[1],q[3];
ry(1.970950699016985) q[1];
ry(0.7107564483617992) q[3];
cx q[1],q[3];
ry(-1.2767520334083937) q[3];
ry(-0.8375716323465907) q[5];
cx q[3],q[5];
ry(2.3595425116957935) q[3];
ry(-2.724249860480626) q[5];
cx q[3],q[5];
ry(-1.634025951983026) q[5];
ry(-0.33808140859245955) q[7];
cx q[5],q[7];
ry(0.2288098727181809) q[5];
ry(-2.461343248852731) q[7];
cx q[5],q[7];
ry(2.6420154221208967) q[0];
ry(1.7505554215191437) q[3];
cx q[0],q[3];
ry(-0.36031937016369897) q[0];
ry(0.42884703476508523) q[3];
cx q[0],q[3];
ry(-2.8068364319365715) q[1];
ry(2.813268636482856) q[2];
cx q[1],q[2];
ry(-3.1408526735969677) q[1];
ry(3.012874352361679) q[2];
cx q[1],q[2];
ry(2.3112717212354443) q[2];
ry(-3.0418103951730417) q[5];
cx q[2],q[5];
ry(-1.8592045346582782) q[2];
ry(1.9850472228862421) q[5];
cx q[2],q[5];
ry(0.14509237099092065) q[3];
ry(-1.5022452559684747) q[4];
cx q[3],q[4];
ry(2.877754956946119) q[3];
ry(-1.4764680141916182) q[4];
cx q[3],q[4];
ry(1.9765128829460084) q[4];
ry(2.806669179181679) q[7];
cx q[4],q[7];
ry(-1.9085911810500744) q[4];
ry(-2.5117881383644134) q[7];
cx q[4],q[7];
ry(-0.6501099927522962) q[5];
ry(-1.6715814685683652) q[6];
cx q[5],q[6];
ry(3.040515079376712) q[5];
ry(-1.7128600916641628) q[6];
cx q[5],q[6];
ry(1.5671058475186948) q[0];
ry(-0.41744788660391585) q[1];
cx q[0],q[1];
ry(2.31620340409929) q[0];
ry(-2.168434394901351) q[1];
cx q[0],q[1];
ry(2.8820949593362006) q[2];
ry(-1.939768783317721) q[3];
cx q[2],q[3];
ry(-1.0156130602232842) q[2];
ry(-1.7829454273672534) q[3];
cx q[2],q[3];
ry(-1.0701839440744232) q[4];
ry(-1.6922735319828082) q[5];
cx q[4],q[5];
ry(-1.9987038336801648) q[4];
ry(-2.2541749384841827) q[5];
cx q[4],q[5];
ry(-2.7718527046886297) q[6];
ry(-0.022675064972518758) q[7];
cx q[6],q[7];
ry(-1.396373732500414) q[6];
ry(-0.9800862860650552) q[7];
cx q[6],q[7];
ry(0.6602024593576828) q[0];
ry(1.4403403994021928) q[2];
cx q[0],q[2];
ry(-2.5810145912383544) q[0];
ry(1.8997202008084646) q[2];
cx q[0],q[2];
ry(-1.373150936567738) q[2];
ry(0.39399257762599804) q[4];
cx q[2],q[4];
ry(0.9633442927077188) q[2];
ry(-2.7689216298305985) q[4];
cx q[2],q[4];
ry(0.17159774729403932) q[4];
ry(-0.2499116952318235) q[6];
cx q[4],q[6];
ry(-2.087943704477724) q[4];
ry(1.2533299800895072) q[6];
cx q[4],q[6];
ry(0.7267567052080192) q[1];
ry(2.575966833650135) q[3];
cx q[1],q[3];
ry(0.22975136446225974) q[1];
ry(0.9693051174622583) q[3];
cx q[1],q[3];
ry(-2.697767367509498) q[3];
ry(1.039045575678431) q[5];
cx q[3],q[5];
ry(2.162336057724693) q[3];
ry(2.3717866768428872) q[5];
cx q[3],q[5];
ry(2.606095008675508) q[5];
ry(2.679756420620255) q[7];
cx q[5],q[7];
ry(2.7775026197127106) q[5];
ry(0.6017511777194106) q[7];
cx q[5],q[7];
ry(-2.561237527471175) q[0];
ry(-1.28858233322336) q[3];
cx q[0],q[3];
ry(-2.288948728872107) q[0];
ry(2.2456942612420967) q[3];
cx q[0],q[3];
ry(-2.8528458109140717) q[1];
ry(-0.059160086453938915) q[2];
cx q[1],q[2];
ry(2.8331351672283365) q[1];
ry(1.4201294041377694) q[2];
cx q[1],q[2];
ry(-1.74125433365619) q[2];
ry(-2.8146416075037974) q[5];
cx q[2],q[5];
ry(0.9951303512486175) q[2];
ry(1.0717680036662962) q[5];
cx q[2],q[5];
ry(-1.1643690791329497) q[3];
ry(-0.5569221383576518) q[4];
cx q[3],q[4];
ry(-0.017612615373241724) q[3];
ry(1.6023235192648613) q[4];
cx q[3],q[4];
ry(-2.3613181300001167) q[4];
ry(-2.496739661715163) q[7];
cx q[4],q[7];
ry(0.6016469545892749) q[4];
ry(-2.44577395023081) q[7];
cx q[4],q[7];
ry(1.0898165917205473) q[5];
ry(0.8782538615400072) q[6];
cx q[5],q[6];
ry(-1.032896078660089) q[5];
ry(1.1271391912150115) q[6];
cx q[5],q[6];
ry(-2.1102644973592035) q[0];
ry(0.3515835641884124) q[1];
cx q[0],q[1];
ry(1.2399376959310409) q[0];
ry(-2.820244806441241) q[1];
cx q[0],q[1];
ry(-1.814636553358549) q[2];
ry(-1.2225258167167954) q[3];
cx q[2],q[3];
ry(-2.895054388525861) q[2];
ry(-1.9226530634991117) q[3];
cx q[2],q[3];
ry(0.05694824262632814) q[4];
ry(-2.843920755467512) q[5];
cx q[4],q[5];
ry(-0.9751879222665687) q[4];
ry(1.8886969611184812) q[5];
cx q[4],q[5];
ry(-0.7832264569160772) q[6];
ry(1.1877782810829272) q[7];
cx q[6],q[7];
ry(1.525886803887757) q[6];
ry(1.385066204528888) q[7];
cx q[6],q[7];
ry(-2.5884651781186547) q[0];
ry(-2.421832594539909) q[2];
cx q[0],q[2];
ry(0.8392923771325763) q[0];
ry(-2.501773095220466) q[2];
cx q[0],q[2];
ry(-1.3251332849924444) q[2];
ry(-1.7523317204467321) q[4];
cx q[2],q[4];
ry(-1.9017366138621121) q[2];
ry(0.034283722177774756) q[4];
cx q[2],q[4];
ry(1.038909929593702) q[4];
ry(2.786796105219634) q[6];
cx q[4],q[6];
ry(0.526024039163075) q[4];
ry(2.0519698787334644) q[6];
cx q[4],q[6];
ry(-2.4301091546844007) q[1];
ry(1.407810347049823) q[3];
cx q[1],q[3];
ry(2.469814745914417) q[1];
ry(-2.437045736016985) q[3];
cx q[1],q[3];
ry(-2.2332828561383504) q[3];
ry(1.077828559591655) q[5];
cx q[3],q[5];
ry(0.714113174678026) q[3];
ry(1.5794666346304682) q[5];
cx q[3],q[5];
ry(2.740506672611302) q[5];
ry(-1.8231093050662979) q[7];
cx q[5],q[7];
ry(1.6044747462190145) q[5];
ry(3.131330456725714) q[7];
cx q[5],q[7];
ry(-2.7344448408931714) q[0];
ry(-1.9884379159555927) q[3];
cx q[0],q[3];
ry(1.1617248986765774) q[0];
ry(0.016462172458305) q[3];
cx q[0],q[3];
ry(0.3905257415114267) q[1];
ry(0.37657507817836056) q[2];
cx q[1],q[2];
ry(1.8130525365092467) q[1];
ry(2.948543528674158) q[2];
cx q[1],q[2];
ry(-0.29093557143789295) q[2];
ry(2.3908675445357037) q[5];
cx q[2],q[5];
ry(1.5801749501328397) q[2];
ry(-0.7951972506678741) q[5];
cx q[2],q[5];
ry(1.9399303036501836) q[3];
ry(2.5849019135367515) q[4];
cx q[3],q[4];
ry(3.0186572577672983) q[3];
ry(3.0710594909317246) q[4];
cx q[3],q[4];
ry(-0.12759234233355699) q[4];
ry(-2.996837148254385) q[7];
cx q[4],q[7];
ry(1.6627459439214238) q[4];
ry(1.6050862405359014) q[7];
cx q[4],q[7];
ry(1.9236370857683618) q[5];
ry(-0.6402956267342468) q[6];
cx q[5],q[6];
ry(1.5453471036152466) q[5];
ry(1.9753500346646384) q[6];
cx q[5],q[6];
ry(-1.6052653494620488) q[0];
ry(1.029181223431472) q[1];
cx q[0],q[1];
ry(-0.7091987532862816) q[0];
ry(0.4124930026279445) q[1];
cx q[0],q[1];
ry(-3.0961787435388604) q[2];
ry(-0.9062919936636843) q[3];
cx q[2],q[3];
ry(2.685879726305367) q[2];
ry(-3.037590593523079) q[3];
cx q[2],q[3];
ry(2.8528321866212467) q[4];
ry(-0.7102971337484678) q[5];
cx q[4],q[5];
ry(-1.8846456552756754) q[4];
ry(3.085577643056949) q[5];
cx q[4],q[5];
ry(-1.8390346545721743) q[6];
ry(0.2545614565701329) q[7];
cx q[6],q[7];
ry(1.4548245583305421) q[6];
ry(-2.1852056613546926) q[7];
cx q[6],q[7];
ry(0.8782751142612675) q[0];
ry(-2.77867531015522) q[2];
cx q[0],q[2];
ry(1.920162651052759) q[0];
ry(2.2918288954969017) q[2];
cx q[0],q[2];
ry(1.8804567469585927) q[2];
ry(2.647423767154378) q[4];
cx q[2],q[4];
ry(-1.7080096549488653) q[2];
ry(2.4101848366169873) q[4];
cx q[2],q[4];
ry(-3.1228077182987555) q[4];
ry(1.4443046352340956) q[6];
cx q[4],q[6];
ry(-0.9476768230272157) q[4];
ry(3.0046376395361083) q[6];
cx q[4],q[6];
ry(0.34561026462587296) q[1];
ry(-1.3483416474755467) q[3];
cx q[1],q[3];
ry(1.2706944012385772) q[1];
ry(-1.0847308897189312) q[3];
cx q[1],q[3];
ry(1.5658058035208877) q[3];
ry(2.8162160975453685) q[5];
cx q[3],q[5];
ry(1.5917700148015113) q[3];
ry(0.3821877075333948) q[5];
cx q[3],q[5];
ry(-2.3290583842507684) q[5];
ry(-0.8144788139158166) q[7];
cx q[5],q[7];
ry(-2.569950013383849) q[5];
ry(-2.058016270080051) q[7];
cx q[5],q[7];
ry(0.3832517108600406) q[0];
ry(-2.970969352363533) q[3];
cx q[0],q[3];
ry(-2.236424932028546) q[0];
ry(1.8131047968855878) q[3];
cx q[0],q[3];
ry(-2.555146116821958) q[1];
ry(-2.820006248245584) q[2];
cx q[1],q[2];
ry(-2.6423765553334286) q[1];
ry(1.1457519777447016) q[2];
cx q[1],q[2];
ry(-2.663185033635586) q[2];
ry(-1.9654819414740334) q[5];
cx q[2],q[5];
ry(0.375766803508463) q[2];
ry(-1.0152653236486344) q[5];
cx q[2],q[5];
ry(1.975925651237155) q[3];
ry(0.9290057200084362) q[4];
cx q[3],q[4];
ry(1.3993428972038946) q[3];
ry(-0.9411727854226708) q[4];
cx q[3],q[4];
ry(-0.5151054124877028) q[4];
ry(1.4831422715108917) q[7];
cx q[4],q[7];
ry(-0.996160791733243) q[4];
ry(-1.9080750335772318) q[7];
cx q[4],q[7];
ry(-0.08839386085557502) q[5];
ry(2.032124156243438) q[6];
cx q[5],q[6];
ry(-0.23368694072975368) q[5];
ry(1.6030062823297815) q[6];
cx q[5],q[6];
ry(-1.7082122626386533) q[0];
ry(0.029865027630173735) q[1];
cx q[0],q[1];
ry(0.4116902752311238) q[0];
ry(2.2262198570553933) q[1];
cx q[0],q[1];
ry(-2.0246618241923313) q[2];
ry(3.0275085385310065) q[3];
cx q[2],q[3];
ry(-1.6565190592129229) q[2];
ry(-2.942168122822145) q[3];
cx q[2],q[3];
ry(-1.8410010811045732) q[4];
ry(-0.8390438543146198) q[5];
cx q[4],q[5];
ry(0.794189301078841) q[4];
ry(0.1488179904494782) q[5];
cx q[4],q[5];
ry(0.48725903823138594) q[6];
ry(-1.308962893152545) q[7];
cx q[6],q[7];
ry(-0.31259734473607004) q[6];
ry(-1.6472545232774245) q[7];
cx q[6],q[7];
ry(3.1336782226557616) q[0];
ry(-1.3488300689624424) q[2];
cx q[0],q[2];
ry(1.0996927730265775) q[0];
ry(-0.5773460972236828) q[2];
cx q[0],q[2];
ry(2.7495417253554764) q[2];
ry(2.078069241620098) q[4];
cx q[2],q[4];
ry(-0.8552625626758088) q[2];
ry(-2.9684481321167886) q[4];
cx q[2],q[4];
ry(-0.20136617899454276) q[4];
ry(-1.894827628808532) q[6];
cx q[4],q[6];
ry(0.08263352182771308) q[4];
ry(1.8185512543109639) q[6];
cx q[4],q[6];
ry(1.7955928040677254) q[1];
ry(-2.728744663213679) q[3];
cx q[1],q[3];
ry(-1.393297851750871) q[1];
ry(0.48503616135864464) q[3];
cx q[1],q[3];
ry(-1.9667884854130693) q[3];
ry(1.2905799132826221) q[5];
cx q[3],q[5];
ry(1.4034166795356944) q[3];
ry(-1.2574569999717367) q[5];
cx q[3],q[5];
ry(-1.8755355172395876) q[5];
ry(2.7547229756471943) q[7];
cx q[5],q[7];
ry(-1.2498623324873488) q[5];
ry(0.8387513030111339) q[7];
cx q[5],q[7];
ry(-0.4288887932474381) q[0];
ry(2.23969584038568) q[3];
cx q[0],q[3];
ry(-2.7816782540262315) q[0];
ry(1.3737943881402064) q[3];
cx q[0],q[3];
ry(-0.8125638598285888) q[1];
ry(2.467635149857353) q[2];
cx q[1],q[2];
ry(0.18732302372635193) q[1];
ry(1.1139359571738368) q[2];
cx q[1],q[2];
ry(-0.009693323988271628) q[2];
ry(-2.9406005106379047) q[5];
cx q[2],q[5];
ry(2.7881882302946503) q[2];
ry(-0.06593979488101538) q[5];
cx q[2],q[5];
ry(-2.8641697254827685) q[3];
ry(-1.0703543804727111) q[4];
cx q[3],q[4];
ry(-1.523047858678825) q[3];
ry(1.0623114742973685) q[4];
cx q[3],q[4];
ry(2.3156804495001264) q[4];
ry(-0.8122388366414327) q[7];
cx q[4],q[7];
ry(1.5382750412635042) q[4];
ry(3.071070500905663) q[7];
cx q[4],q[7];
ry(-0.874042002807748) q[5];
ry(-2.963849014409531) q[6];
cx q[5],q[6];
ry(-1.0478558110164577) q[5];
ry(-1.9567424960619348) q[6];
cx q[5],q[6];
ry(-1.9494303234648969) q[0];
ry(-2.2448130398486734) q[1];
cx q[0],q[1];
ry(-1.7945850073363872) q[0];
ry(-0.33786968654569455) q[1];
cx q[0],q[1];
ry(-2.6933001811852764) q[2];
ry(-0.9834184798838287) q[3];
cx q[2],q[3];
ry(1.6257194546630664) q[2];
ry(-1.9033838775858871) q[3];
cx q[2],q[3];
ry(-2.980945783189556) q[4];
ry(-2.706913095254049) q[5];
cx q[4],q[5];
ry(2.851251667616619) q[4];
ry(0.338965791848131) q[5];
cx q[4],q[5];
ry(-1.9243502115651456) q[6];
ry(-1.8337962650664679) q[7];
cx q[6],q[7];
ry(-0.045091084270314497) q[6];
ry(-0.7899789853087) q[7];
cx q[6],q[7];
ry(2.673272152080635) q[0];
ry(-1.3371308343944337) q[2];
cx q[0],q[2];
ry(-0.24770569880667637) q[0];
ry(1.3675429429850823) q[2];
cx q[0],q[2];
ry(2.6253224920086202) q[2];
ry(-0.6407721793974854) q[4];
cx q[2],q[4];
ry(-0.023791679007855393) q[2];
ry(1.6806632006806934) q[4];
cx q[2],q[4];
ry(-0.05744758870162947) q[4];
ry(-1.7208497948003612) q[6];
cx q[4],q[6];
ry(1.4100464000750241) q[4];
ry(-1.304824541075467) q[6];
cx q[4],q[6];
ry(2.330247969283089) q[1];
ry(-0.5555944681053354) q[3];
cx q[1],q[3];
ry(-0.7817917873349165) q[1];
ry(-3.1165067985977104) q[3];
cx q[1],q[3];
ry(1.6252488114870927) q[3];
ry(3.1219422150752183) q[5];
cx q[3],q[5];
ry(1.451675199962995) q[3];
ry(2.633704124750214) q[5];
cx q[3],q[5];
ry(-0.7449879570147524) q[5];
ry(1.3101336722942947) q[7];
cx q[5],q[7];
ry(-2.891380948697484) q[5];
ry(-1.8196514099293437) q[7];
cx q[5],q[7];
ry(2.4068799511442127) q[0];
ry(2.390178376195628) q[3];
cx q[0],q[3];
ry(0.6013931464480207) q[0];
ry(0.4511886839011652) q[3];
cx q[0],q[3];
ry(-1.0619176749007373) q[1];
ry(-0.4975762962780807) q[2];
cx q[1],q[2];
ry(-1.534815436533763) q[1];
ry(-2.8271968680839716) q[2];
cx q[1],q[2];
ry(2.94484709003591) q[2];
ry(-1.987070683648895) q[5];
cx q[2],q[5];
ry(1.5837757393805572) q[2];
ry(-1.6021815264235821) q[5];
cx q[2],q[5];
ry(2.6218898028676856) q[3];
ry(-0.5167898461395761) q[4];
cx q[3],q[4];
ry(0.7801384022349609) q[3];
ry(-1.0157658119847215) q[4];
cx q[3],q[4];
ry(1.8781580651246248) q[4];
ry(-2.5326732086031445) q[7];
cx q[4],q[7];
ry(-0.8328551251874305) q[4];
ry(0.5963623410740161) q[7];
cx q[4],q[7];
ry(-2.1386498068845063) q[5];
ry(1.1803419349856092) q[6];
cx q[5],q[6];
ry(-0.7165238066607875) q[5];
ry(1.9322492796854311) q[6];
cx q[5],q[6];
ry(2.2905795384527825) q[0];
ry(-2.485450120105011) q[1];
cx q[0],q[1];
ry(-1.3699415753385529) q[0];
ry(-3.044410136038088) q[1];
cx q[0],q[1];
ry(-1.1846438856713704) q[2];
ry(-0.49655981884955475) q[3];
cx q[2],q[3];
ry(2.8616688644242796) q[2];
ry(-0.15852305481593762) q[3];
cx q[2],q[3];
ry(-2.9202379165272077) q[4];
ry(-1.8731174926935603) q[5];
cx q[4],q[5];
ry(0.8663320901816602) q[4];
ry(0.8624226480874473) q[5];
cx q[4],q[5];
ry(0.5744716790305969) q[6];
ry(0.7030023782543808) q[7];
cx q[6],q[7];
ry(1.6760330211785162) q[6];
ry(-2.3644304755070458) q[7];
cx q[6],q[7];
ry(1.4568228577864915) q[0];
ry(0.2955468291867812) q[2];
cx q[0],q[2];
ry(1.2349753540745816) q[0];
ry(-2.1442012291820554) q[2];
cx q[0],q[2];
ry(-2.10681072470899) q[2];
ry(-2.1011752149180314) q[4];
cx q[2],q[4];
ry(-0.4448668865583168) q[2];
ry(1.0147407350210695) q[4];
cx q[2],q[4];
ry(-2.8503183542604362) q[4];
ry(1.140708885762339) q[6];
cx q[4],q[6];
ry(2.7994709448438915) q[4];
ry(0.4762255145597374) q[6];
cx q[4],q[6];
ry(-2.7767086266539014) q[1];
ry(2.5813150718913005) q[3];
cx q[1],q[3];
ry(-1.4901835238292176) q[1];
ry(-2.7831066467014334) q[3];
cx q[1],q[3];
ry(-2.5151675496332317) q[3];
ry(1.1166510198462136) q[5];
cx q[3],q[5];
ry(-1.0297963748374386) q[3];
ry(-2.170324150868735) q[5];
cx q[3],q[5];
ry(0.3278120670589999) q[5];
ry(-1.6050687650651485) q[7];
cx q[5],q[7];
ry(0.18687257150765232) q[5];
ry(1.4070104868979065) q[7];
cx q[5],q[7];
ry(0.11106302544196313) q[0];
ry(-1.1932722049389044) q[3];
cx q[0],q[3];
ry(1.6490794945164702) q[0];
ry(-0.48847083686368453) q[3];
cx q[0],q[3];
ry(-2.0032441051709187) q[1];
ry(0.3458242823482403) q[2];
cx q[1],q[2];
ry(0.3849653452978333) q[1];
ry(1.0522954042054113) q[2];
cx q[1],q[2];
ry(-2.03623935397256) q[2];
ry(-0.42028036579544636) q[5];
cx q[2],q[5];
ry(2.0352482121064144) q[2];
ry(1.7403132030122537) q[5];
cx q[2],q[5];
ry(0.4689478133777865) q[3];
ry(-2.2339490387075482) q[4];
cx q[3],q[4];
ry(0.9113996559423061) q[3];
ry(0.45013040604813737) q[4];
cx q[3],q[4];
ry(-2.5211353144998148) q[4];
ry(1.8523235373278832) q[7];
cx q[4],q[7];
ry(-0.10951072936304874) q[4];
ry(-2.3044259655204806) q[7];
cx q[4],q[7];
ry(-0.9017862963128547) q[5];
ry(-2.8292441648550097) q[6];
cx q[5],q[6];
ry(2.951161634880354) q[5];
ry(-2.837351092949128) q[6];
cx q[5],q[6];
ry(2.5552873225044097) q[0];
ry(-1.94858586875438) q[1];
cx q[0],q[1];
ry(-2.850976695644847) q[0];
ry(-2.031212442173534) q[1];
cx q[0],q[1];
ry(0.9802627297442676) q[2];
ry(1.2777431558065258) q[3];
cx q[2],q[3];
ry(-0.5294031671421827) q[2];
ry(-2.1497177927314732) q[3];
cx q[2],q[3];
ry(-0.15649694595892605) q[4];
ry(-0.8154956868396033) q[5];
cx q[4],q[5];
ry(-2.5361932797945617) q[4];
ry(0.7229231245087647) q[5];
cx q[4],q[5];
ry(0.13091653306759002) q[6];
ry(0.25088690832431604) q[7];
cx q[6],q[7];
ry(2.1540071073573905) q[6];
ry(-0.9457132414508442) q[7];
cx q[6],q[7];
ry(1.412809123275147) q[0];
ry(-0.3168958260219751) q[2];
cx q[0],q[2];
ry(2.907159130691828) q[0];
ry(0.5369623496643551) q[2];
cx q[0],q[2];
ry(1.6925147015614916) q[2];
ry(0.8836282894907237) q[4];
cx q[2],q[4];
ry(-0.6891268425635113) q[2];
ry(-2.8363905323609058) q[4];
cx q[2],q[4];
ry(-1.0188970779818485) q[4];
ry(2.4500763758970616) q[6];
cx q[4],q[6];
ry(0.43764279647454635) q[4];
ry(-2.2498134130247345) q[6];
cx q[4],q[6];
ry(-1.5885172289208547) q[1];
ry(-0.10209623467802616) q[3];
cx q[1],q[3];
ry(0.6153368781720255) q[1];
ry(1.8838453747849204) q[3];
cx q[1],q[3];
ry(-0.36038597452813903) q[3];
ry(-2.3122793944316866) q[5];
cx q[3],q[5];
ry(0.8874612065251437) q[3];
ry(-2.3283075130320343) q[5];
cx q[3],q[5];
ry(2.3136174492357835) q[5];
ry(-3.0384383506897237) q[7];
cx q[5],q[7];
ry(1.6386194677334636) q[5];
ry(0.16366686571532085) q[7];
cx q[5],q[7];
ry(1.8992329204212242) q[0];
ry(1.845397782361176) q[3];
cx q[0],q[3];
ry(-1.1934070582639444) q[0];
ry(1.526677945088086) q[3];
cx q[0],q[3];
ry(1.5976200073629716) q[1];
ry(-1.7322144928704368) q[2];
cx q[1],q[2];
ry(-1.7211912399297404) q[1];
ry(-1.1147655208348421) q[2];
cx q[1],q[2];
ry(-0.7848875459930609) q[2];
ry(1.049872180748485) q[5];
cx q[2],q[5];
ry(0.21771354958926636) q[2];
ry(2.634803993988013) q[5];
cx q[2],q[5];
ry(-2.0193596255732453) q[3];
ry(1.8635607695720136) q[4];
cx q[3],q[4];
ry(2.4535875143364922) q[3];
ry(1.2804332080090448) q[4];
cx q[3],q[4];
ry(-3.1394731374431033) q[4];
ry(-2.369666699889658) q[7];
cx q[4],q[7];
ry(-2.137574883283949) q[4];
ry(-1.8242492182613324) q[7];
cx q[4],q[7];
ry(1.3539403511078296) q[5];
ry(0.9571863263282447) q[6];
cx q[5],q[6];
ry(0.2727434463629751) q[5];
ry(0.5656339032120333) q[6];
cx q[5],q[6];
ry(-1.1763514013473824) q[0];
ry(2.970028008341831) q[1];
cx q[0],q[1];
ry(3.0604673318019135) q[0];
ry(-0.8747409076846077) q[1];
cx q[0],q[1];
ry(-2.986887656625349) q[2];
ry(1.356830533480381) q[3];
cx q[2],q[3];
ry(2.8476950882806658) q[2];
ry(2.5303416575570754) q[3];
cx q[2],q[3];
ry(2.314483673594941) q[4];
ry(-1.0071784263138372) q[5];
cx q[4],q[5];
ry(-2.448021501355671) q[4];
ry(0.15046348121483621) q[5];
cx q[4],q[5];
ry(0.03180613744063325) q[6];
ry(0.8918790447660598) q[7];
cx q[6],q[7];
ry(2.9111923198386624) q[6];
ry(-3.1224061412599955) q[7];
cx q[6],q[7];
ry(-1.8596101453501237) q[0];
ry(0.46339162345739376) q[2];
cx q[0],q[2];
ry(-1.4879044239199615) q[0];
ry(-1.4133273695850077) q[2];
cx q[0],q[2];
ry(2.2868861978628616) q[2];
ry(-2.7506923119004583) q[4];
cx q[2],q[4];
ry(0.7838627251413338) q[2];
ry(0.9491058197334263) q[4];
cx q[2],q[4];
ry(-2.225467606238174) q[4];
ry(1.7065074133702856) q[6];
cx q[4],q[6];
ry(2.5438844367385496) q[4];
ry(-1.8281992465788557) q[6];
cx q[4],q[6];
ry(2.6687527358140732) q[1];
ry(-2.1312983355209982) q[3];
cx q[1],q[3];
ry(-2.199300254359155) q[1];
ry(-0.540832331044739) q[3];
cx q[1],q[3];
ry(-1.412208527518917) q[3];
ry(1.5231798649359283) q[5];
cx q[3],q[5];
ry(-0.503461691658103) q[3];
ry(-1.5967516039102196) q[5];
cx q[3],q[5];
ry(0.13370576010575963) q[5];
ry(2.44835290688823) q[7];
cx q[5],q[7];
ry(-1.4420390691098337) q[5];
ry(-1.021497330743727) q[7];
cx q[5],q[7];
ry(2.839634698489778) q[0];
ry(-0.016835493761719) q[3];
cx q[0],q[3];
ry(-0.3899617241812101) q[0];
ry(2.6012315391438467) q[3];
cx q[0],q[3];
ry(2.274307060708873) q[1];
ry(1.5019652482254262) q[2];
cx q[1],q[2];
ry(-1.9310505989932611) q[1];
ry(-2.062284129756101) q[2];
cx q[1],q[2];
ry(0.40741551955947836) q[2];
ry(3.070816889929138) q[5];
cx q[2],q[5];
ry(1.5011895470531458) q[2];
ry(2.5563099488388996) q[5];
cx q[2],q[5];
ry(2.17986623004208) q[3];
ry(-0.43488218983380733) q[4];
cx q[3],q[4];
ry(-1.4150254374400202) q[3];
ry(0.9762753021120909) q[4];
cx q[3],q[4];
ry(-0.36140392080914463) q[4];
ry(-2.4743754312522874) q[7];
cx q[4],q[7];
ry(2.6145754532379635) q[4];
ry(-2.8235137928103597) q[7];
cx q[4],q[7];
ry(-2.437336048907717) q[5];
ry(-1.9628607151407138) q[6];
cx q[5],q[6];
ry(-3.075951452446965) q[5];
ry(0.3293785023090293) q[6];
cx q[5],q[6];
ry(1.759508542821747) q[0];
ry(2.8554730026508257) q[1];
cx q[0],q[1];
ry(-2.36395767528441) q[0];
ry(-2.8789911836256135) q[1];
cx q[0],q[1];
ry(1.0390205609055716) q[2];
ry(1.9994782739476413) q[3];
cx q[2],q[3];
ry(-1.016448378942746) q[2];
ry(-0.3465760636967304) q[3];
cx q[2],q[3];
ry(1.7297137739996922) q[4];
ry(-1.5806371739202891) q[5];
cx q[4],q[5];
ry(0.2015395648869435) q[4];
ry(3.0161495524678528) q[5];
cx q[4],q[5];
ry(-0.27422348900342275) q[6];
ry(2.7718956328228384) q[7];
cx q[6],q[7];
ry(2.4396414830219895) q[6];
ry(1.8250394548866682) q[7];
cx q[6],q[7];
ry(3.049834040788575) q[0];
ry(1.8689329514392314) q[2];
cx q[0],q[2];
ry(-0.29775502385872343) q[0];
ry(-2.20891658481238) q[2];
cx q[0],q[2];
ry(-0.8141886874389934) q[2];
ry(-0.550418276737286) q[4];
cx q[2],q[4];
ry(2.264001746815526) q[2];
ry(0.7807627043121537) q[4];
cx q[2],q[4];
ry(-1.9670645600606358) q[4];
ry(1.2586527188443153) q[6];
cx q[4],q[6];
ry(2.467332898669148) q[4];
ry(2.2052361410769192) q[6];
cx q[4],q[6];
ry(-0.670111824100243) q[1];
ry(-0.8058606048434598) q[3];
cx q[1],q[3];
ry(-3.015536085171066) q[1];
ry(-0.34199955188018033) q[3];
cx q[1],q[3];
ry(2.759581650638206) q[3];
ry(-1.2048812651319083) q[5];
cx q[3],q[5];
ry(-1.7102305952902856) q[3];
ry(-0.182322613911273) q[5];
cx q[3],q[5];
ry(-3.05506846510656) q[5];
ry(0.9897997546269304) q[7];
cx q[5],q[7];
ry(1.9903396039607175) q[5];
ry(0.9328720826164585) q[7];
cx q[5],q[7];
ry(1.687027757258597) q[0];
ry(-1.5649743812863204) q[3];
cx q[0],q[3];
ry(-2.7793927696944487) q[0];
ry(-1.3782984120404311) q[3];
cx q[0],q[3];
ry(0.4797498509071527) q[1];
ry(0.43165718309295364) q[2];
cx q[1],q[2];
ry(-1.1327059831485429) q[1];
ry(-1.977886804194227) q[2];
cx q[1],q[2];
ry(0.523010304312856) q[2];
ry(-2.35667010381363) q[5];
cx q[2],q[5];
ry(3.0384557049295546) q[2];
ry(-0.7992365614848307) q[5];
cx q[2],q[5];
ry(-0.8566317087468766) q[3];
ry(1.830621424900019) q[4];
cx q[3],q[4];
ry(0.5648575922546192) q[3];
ry(-0.4684773141805059) q[4];
cx q[3],q[4];
ry(-2.8271409407191546) q[4];
ry(-2.661847886826144) q[7];
cx q[4],q[7];
ry(1.4395929937112197) q[4];
ry(-0.42950914247464306) q[7];
cx q[4],q[7];
ry(2.22792536170117) q[5];
ry(-2.3011921633055232) q[6];
cx q[5],q[6];
ry(-0.028443994204814693) q[5];
ry(-3.014053564766258) q[6];
cx q[5],q[6];
ry(-0.054089873193411186) q[0];
ry(-1.511946674308466) q[1];
cx q[0],q[1];
ry(2.0609991452195295) q[0];
ry(-1.0626748833294428) q[1];
cx q[0],q[1];
ry(2.2983824680810874) q[2];
ry(-0.48722570989694897) q[3];
cx q[2],q[3];
ry(0.2791307517536412) q[2];
ry(2.906124795241571) q[3];
cx q[2],q[3];
ry(0.5483388829446207) q[4];
ry(-1.095929296363292) q[5];
cx q[4],q[5];
ry(-0.038486670008308445) q[4];
ry(-1.2270985851870873) q[5];
cx q[4],q[5];
ry(0.7233050368021834) q[6];
ry(1.7148890899636662) q[7];
cx q[6],q[7];
ry(2.390278442074559) q[6];
ry(-0.033560659577162116) q[7];
cx q[6],q[7];
ry(0.7081447571999568) q[0];
ry(-2.666653072268766) q[2];
cx q[0],q[2];
ry(-0.8848293398005633) q[0];
ry(1.3095598293281128) q[2];
cx q[0],q[2];
ry(-1.1520389200603702) q[2];
ry(-0.7653890851354141) q[4];
cx q[2],q[4];
ry(-2.3157835933955293) q[2];
ry(2.2895065970426485) q[4];
cx q[2],q[4];
ry(2.959469674907179) q[4];
ry(2.8629098797032753) q[6];
cx q[4],q[6];
ry(2.413994774898126) q[4];
ry(0.5685519444148147) q[6];
cx q[4],q[6];
ry(3.0012367989027133) q[1];
ry(-1.5016359329134017) q[3];
cx q[1],q[3];
ry(1.6203421959479518) q[1];
ry(1.0959565483182656) q[3];
cx q[1],q[3];
ry(1.2612258879457583) q[3];
ry(-0.8053271705967697) q[5];
cx q[3],q[5];
ry(1.310661316956675) q[3];
ry(0.7450534931344269) q[5];
cx q[3],q[5];
ry(-3.1173879964688944) q[5];
ry(2.8635534410152315) q[7];
cx q[5],q[7];
ry(-3.008457329163671) q[5];
ry(-0.8296258759151806) q[7];
cx q[5],q[7];
ry(-0.5815166681049384) q[0];
ry(-0.8140097167057113) q[3];
cx q[0],q[3];
ry(0.7182280673476082) q[0];
ry(0.6215248278114407) q[3];
cx q[0],q[3];
ry(2.069593912204601) q[1];
ry(-1.8472815761092347) q[2];
cx q[1],q[2];
ry(-3.1249673662402992) q[1];
ry(-1.0047186326656883) q[2];
cx q[1],q[2];
ry(0.04417763393014989) q[2];
ry(-2.3802159724274974) q[5];
cx q[2],q[5];
ry(-0.9093769868952082) q[2];
ry(-2.809498168227316) q[5];
cx q[2],q[5];
ry(-1.7880499758732804) q[3];
ry(2.072462726905972) q[4];
cx q[3],q[4];
ry(-2.627067723257286) q[3];
ry(1.1917132076473482) q[4];
cx q[3],q[4];
ry(0.5734836963108709) q[4];
ry(-1.9042922456708453) q[7];
cx q[4],q[7];
ry(-2.0371084569139732) q[4];
ry(-0.9944307143462475) q[7];
cx q[4],q[7];
ry(2.7195066983990306) q[5];
ry(-2.0438592931683406) q[6];
cx q[5],q[6];
ry(2.571193073661974) q[5];
ry(0.8480096744321631) q[6];
cx q[5],q[6];
ry(-1.4532824213466893) q[0];
ry(-1.3747008815030313) q[1];
cx q[0],q[1];
ry(-0.7760634276379134) q[0];
ry(0.8015952562325827) q[1];
cx q[0],q[1];
ry(-2.647389218075307) q[2];
ry(1.5308985173014564) q[3];
cx q[2],q[3];
ry(-3.0135001656375344) q[2];
ry(-1.6528540619843393) q[3];
cx q[2],q[3];
ry(1.5507894683897427) q[4];
ry(-0.9288636119046112) q[5];
cx q[4],q[5];
ry(0.09711684998774482) q[4];
ry(-0.2863514610589253) q[5];
cx q[4],q[5];
ry(2.700012043439552) q[6];
ry(-2.041025803609865) q[7];
cx q[6],q[7];
ry(0.09413852095938802) q[6];
ry(1.8618715462359605) q[7];
cx q[6],q[7];
ry(2.2461094526360306) q[0];
ry(1.231635860688436) q[2];
cx q[0],q[2];
ry(-2.860555549726335) q[0];
ry(-0.30667868881488225) q[2];
cx q[0],q[2];
ry(3.103940841327584) q[2];
ry(-0.33467738739493225) q[4];
cx q[2],q[4];
ry(1.859807058044301) q[2];
ry(1.2227169793571957) q[4];
cx q[2],q[4];
ry(1.934775608419225) q[4];
ry(2.3085512088826827) q[6];
cx q[4],q[6];
ry(2.203946488118512) q[4];
ry(1.9533111449682323) q[6];
cx q[4],q[6];
ry(1.7305213866162896) q[1];
ry(-1.0132992585787908) q[3];
cx q[1],q[3];
ry(1.0832469031463228) q[1];
ry(-1.2655747551315573) q[3];
cx q[1],q[3];
ry(0.030102273273808322) q[3];
ry(-1.1542793199135213) q[5];
cx q[3],q[5];
ry(2.4553560351614636) q[3];
ry(0.11327796537920179) q[5];
cx q[3],q[5];
ry(-0.373101566369602) q[5];
ry(0.6591692490367113) q[7];
cx q[5],q[7];
ry(2.610073895523014) q[5];
ry(-0.7780604225510253) q[7];
cx q[5],q[7];
ry(-2.638810597579484) q[0];
ry(-2.702098907345778) q[3];
cx q[0],q[3];
ry(1.7768765021786335) q[0];
ry(0.8543957405705773) q[3];
cx q[0],q[3];
ry(1.0713950703993849) q[1];
ry(-0.9694037427927826) q[2];
cx q[1],q[2];
ry(-2.4799398823001146) q[1];
ry(0.7407797014714541) q[2];
cx q[1],q[2];
ry(-0.5762235122249404) q[2];
ry(-1.9299041053368633) q[5];
cx q[2],q[5];
ry(2.0822688136197023) q[2];
ry(-0.30318729710999753) q[5];
cx q[2],q[5];
ry(-0.26167448626855266) q[3];
ry(0.32999825311368286) q[4];
cx q[3],q[4];
ry(2.899408240892789) q[3];
ry(-0.8428469583945376) q[4];
cx q[3],q[4];
ry(-0.4192931915474047) q[4];
ry(-0.13538074338266348) q[7];
cx q[4],q[7];
ry(2.3716646117345284) q[4];
ry(-1.3921947401371773) q[7];
cx q[4],q[7];
ry(0.6957095027334111) q[5];
ry(2.8871420677113244) q[6];
cx q[5],q[6];
ry(2.406516409619701) q[5];
ry(-0.215703293403406) q[6];
cx q[5],q[6];
ry(-2.732971367937821) q[0];
ry(-1.4615469985180412) q[1];
cx q[0],q[1];
ry(1.80220240901079) q[0];
ry(-0.5020378038068718) q[1];
cx q[0],q[1];
ry(0.8531293783259857) q[2];
ry(-0.3039504122720157) q[3];
cx q[2],q[3];
ry(-3.073373576969217) q[2];
ry(-1.9223071606098205) q[3];
cx q[2],q[3];
ry(1.7857148619817051) q[4];
ry(2.8731013430247327) q[5];
cx q[4],q[5];
ry(2.7672476318647856) q[4];
ry(-1.8182242335473449) q[5];
cx q[4],q[5];
ry(-2.5749878190239284) q[6];
ry(-0.21573073463379444) q[7];
cx q[6],q[7];
ry(-0.004439554804974148) q[6];
ry(-0.5653813543345003) q[7];
cx q[6],q[7];
ry(2.4615128575636502) q[0];
ry(0.018676241475408317) q[2];
cx q[0],q[2];
ry(-1.2539853080117025) q[0];
ry(2.340216184307687) q[2];
cx q[0],q[2];
ry(0.6534999261278482) q[2];
ry(2.317950930666206) q[4];
cx q[2],q[4];
ry(-2.061826658486776) q[2];
ry(-0.9035617724669613) q[4];
cx q[2],q[4];
ry(2.51577660044149) q[4];
ry(1.2070974285070077) q[6];
cx q[4],q[6];
ry(1.7865179112637781) q[4];
ry(-1.0797440833353529) q[6];
cx q[4],q[6];
ry(-0.22835130945169893) q[1];
ry(0.013860626944003453) q[3];
cx q[1],q[3];
ry(0.36237090500768654) q[1];
ry(2.858985348211149) q[3];
cx q[1],q[3];
ry(2.0727000963478135) q[3];
ry(2.24942269382662) q[5];
cx q[3],q[5];
ry(-0.8060473605093579) q[3];
ry(2.788603888696) q[5];
cx q[3],q[5];
ry(-2.0980116284056516) q[5];
ry(-0.7358217181492627) q[7];
cx q[5],q[7];
ry(-1.4589635082777077) q[5];
ry(-0.3776948363945811) q[7];
cx q[5],q[7];
ry(3.0314176152803856) q[0];
ry(2.5510657977015447) q[3];
cx q[0],q[3];
ry(-1.1616111351000056) q[0];
ry(0.13362614791491126) q[3];
cx q[0],q[3];
ry(-2.7390075037788373) q[1];
ry(1.6692657858692432) q[2];
cx q[1],q[2];
ry(-0.5322364345702475) q[1];
ry(-3.0585519978241917) q[2];
cx q[1],q[2];
ry(-2.1263031818785487) q[2];
ry(2.9149454918864683) q[5];
cx q[2],q[5];
ry(0.5942973082448635) q[2];
ry(1.874618599809793) q[5];
cx q[2],q[5];
ry(-1.7257036397225356) q[3];
ry(0.948328676182297) q[4];
cx q[3],q[4];
ry(2.0513247297582633) q[3];
ry(-2.843531133464856) q[4];
cx q[3],q[4];
ry(-2.653033347294577) q[4];
ry(3.0153756666065803) q[7];
cx q[4],q[7];
ry(0.6199701356154278) q[4];
ry(-2.956163031507264) q[7];
cx q[4],q[7];
ry(1.8735581967673107) q[5];
ry(2.360345429578543) q[6];
cx q[5],q[6];
ry(1.8512780095597834) q[5];
ry(-3.0595184669036626) q[6];
cx q[5],q[6];
ry(-0.8856721394411355) q[0];
ry(0.32722675792503886) q[1];
ry(-1.315257392005076) q[2];
ry(-1.0866396913575944) q[3];
ry(-3.0858104513035913) q[4];
ry(2.5416234207992803) q[5];
ry(1.2702445422122928) q[6];
ry(-1.0172395691894662) q[7];