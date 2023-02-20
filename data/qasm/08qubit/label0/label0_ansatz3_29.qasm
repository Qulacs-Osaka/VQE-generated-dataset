OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.7519911479251897) q[0];
rz(3.099260884445776) q[0];
ry(-1.023300856553737) q[1];
rz(-0.0064949825048268295) q[1];
ry(2.482654568138774) q[2];
rz(1.6111213623099263) q[2];
ry(-2.104630873142252) q[3];
rz(-0.43448686319269214) q[3];
ry(2.1819779856598838) q[4];
rz(-1.7392171488322958) q[4];
ry(1.8997964035351391) q[5];
rz(0.40721689635831065) q[5];
ry(2.892492577462493) q[6];
rz(0.7742805535135269) q[6];
ry(-0.5041273124849912) q[7];
rz(-1.8442569818009982) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.7262805320364505) q[0];
rz(2.960228498943761) q[0];
ry(2.643583725679117) q[1];
rz(3.06638062529018) q[1];
ry(-2.3621989043855516) q[2];
rz(-3.0586792114687955) q[2];
ry(0.7770195584997676) q[3];
rz(0.026238746905990287) q[3];
ry(1.779948234096221) q[4];
rz(-0.7750626684667717) q[4];
ry(-1.758907081214879) q[5];
rz(2.4415196583994394) q[5];
ry(1.775172562813106) q[6];
rz(2.0430001688435766) q[6];
ry(-1.980147589559959) q[7];
rz(-1.210961509218479) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.456735039612269) q[0];
rz(2.2482276103941308) q[0];
ry(-0.7820117277583565) q[1];
rz(1.179239384858013) q[1];
ry(-0.16051187069469916) q[2];
rz(-1.6146588786651261) q[2];
ry(3.107583042265389) q[3];
rz(-1.366825932618763) q[3];
ry(1.3821316015433558) q[4];
rz(-0.09935046809369488) q[4];
ry(-2.0280667201921005) q[5];
rz(-2.7342298636320894) q[5];
ry(-0.6865921603037277) q[6];
rz(0.24324565928132844) q[6];
ry(1.8681527148909363) q[7];
rz(0.9975592596035091) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.2206198845680216) q[0];
rz(1.083314364081333) q[0];
ry(-1.1819097851691835) q[1];
rz(-0.7683834509355965) q[1];
ry(-2.6305460723300667) q[2];
rz(-1.6268582684850168) q[2];
ry(-2.699698461972011) q[3];
rz(0.14800686531210516) q[3];
ry(-2.5164134927193533) q[4];
rz(0.6526081548530263) q[4];
ry(-1.675941845806122) q[5];
rz(-0.9059889908846573) q[5];
ry(2.655932081016616) q[6];
rz(1.422691757458966) q[6];
ry(0.5655223963993736) q[7];
rz(1.3887899562985853) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-3.0309269358780404) q[0];
rz(-0.4711268400147644) q[0];
ry(-0.6578735671841046) q[1];
rz(-0.9052962526786422) q[1];
ry(2.15769999639475) q[2];
rz(1.9450378003142086) q[2];
ry(0.8549828576180563) q[3];
rz(-1.008993767901563) q[3];
ry(-1.6549636173489584) q[4];
rz(2.383333030158081) q[4];
ry(2.0611477034265886) q[5];
rz(1.6494975788989472) q[5];
ry(-3.0424214785887487) q[6];
rz(1.6469146671028019) q[6];
ry(-0.45377517704522463) q[7];
rz(-1.2130421407991436) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.933422623555983) q[0];
rz(2.5320748962371353) q[0];
ry(1.53065449446295) q[1];
rz(1.2312932708948585) q[1];
ry(-1.43124101770149) q[2];
rz(2.07946247788919) q[2];
ry(-2.755617626633065) q[3];
rz(2.114185412414037) q[3];
ry(-0.08971585575526664) q[4];
rz(2.028201594448076) q[4];
ry(1.2131565994244566) q[5];
rz(-1.1265243308879533) q[5];
ry(2.843223222415152) q[6];
rz(1.9009032182249768) q[6];
ry(0.16324819375394917) q[7];
rz(-2.415758882248973) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.11713556158891707) q[0];
rz(1.9121401125990642) q[0];
ry(3.0948176194411565) q[1];
rz(-1.3190671257731208) q[1];
ry(-1.6583011804690022) q[2];
rz(-0.7390435347055909) q[2];
ry(-2.4764351848331203) q[3];
rz(1.5937136360438056) q[3];
ry(-2.978351389154543) q[4];
rz(-3.1254981702378393) q[4];
ry(-0.8714616277508878) q[5];
rz(2.6325388011798982) q[5];
ry(-0.36793241650774855) q[6];
rz(-3.1223512902626194) q[6];
ry(1.4795089691994943) q[7];
rz(1.0014840007058388) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.7011516187847224) q[0];
rz(2.136867796222372) q[0];
ry(2.3071937249383083) q[1];
rz(-1.5136773135149932) q[1];
ry(-2.5160618588377033) q[2];
rz(0.17055299699529228) q[2];
ry(0.7154775757276353) q[3];
rz(1.6825039693868171) q[3];
ry(-1.8224900438511764) q[4];
rz(-2.495353277338025) q[4];
ry(2.3506390338852485) q[5];
rz(1.7832138829134447) q[5];
ry(-1.519544695763936) q[6];
rz(2.4463967881160147) q[6];
ry(1.0840621797618621) q[7];
rz(1.006181097263604) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.5818398077151278) q[0];
rz(-1.9660494046934496) q[0];
ry(-0.31509964191919343) q[1];
rz(2.112794882907557) q[1];
ry(2.2636642141984398) q[2];
rz(-1.2542340006509491) q[2];
ry(-2.403207707440562) q[3];
rz(0.7301717819382852) q[3];
ry(-1.9374377130467604) q[4];
rz(-2.5789291501808886) q[4];
ry(-1.4128860025216037) q[5];
rz(0.7782331961817874) q[5];
ry(-2.204941366487744) q[6];
rz(-0.996295082141614) q[6];
ry(-1.4173080334595394) q[7];
rz(-2.748329016193648) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.0273119630266563) q[0];
rz(-2.62194313915932) q[0];
ry(-1.0146893267452644) q[1];
rz(-2.5801835009833893) q[1];
ry(3.1052809044509893) q[2];
rz(2.621496134150936) q[2];
ry(-2.3904418798381233) q[3];
rz(-1.2820390357495244) q[3];
ry(-2.5355976253265764) q[4];
rz(1.5221005728906256) q[4];
ry(1.3665384692150582) q[5];
rz(1.870498446944894) q[5];
ry(2.0906953507100363) q[6];
rz(0.8568065603269823) q[6];
ry(-0.7055474470497297) q[7];
rz(-0.0868439749877853) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.6789840062860406) q[0];
rz(2.454100419796434) q[0];
ry(-0.9529989593528251) q[1];
rz(0.5402529349732883) q[1];
ry(-2.5566059638014775) q[2];
rz(0.11140326514566645) q[2];
ry(1.5739213559243002) q[3];
rz(-2.9590481059500577) q[3];
ry(-2.7283996453738752) q[4];
rz(-3.131438087398377) q[4];
ry(1.79497849573748) q[5];
rz(2.668890363511003) q[5];
ry(0.1706093218764435) q[6];
rz(-2.3373870189921186) q[6];
ry(0.3330645240647856) q[7];
rz(-2.2176327688174653) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.968597565323363) q[0];
rz(0.9998429807043808) q[0];
ry(-2.933866093743248) q[1];
rz(-3.059779479961334) q[1];
ry(-0.3491413021758684) q[2];
rz(0.7140932698254803) q[2];
ry(-0.3257626429384235) q[3];
rz(-2.2549090989016656) q[3];
ry(-1.5193215131745326) q[4];
rz(1.143201964380241) q[4];
ry(0.41107524183159644) q[5];
rz(-0.12349933489322316) q[5];
ry(-2.832651961891503) q[6];
rz(1.5041320870927342) q[6];
ry(-0.19929551847084154) q[7];
rz(1.9315848126790955) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.99037444394484) q[0];
rz(2.602389575805665) q[0];
ry(-1.5615263186353625) q[1];
rz(0.7681912522797353) q[1];
ry(0.9651695982951737) q[2];
rz(2.503988189759124) q[2];
ry(1.8454934997551389) q[3];
rz(3.1043093296073914) q[3];
ry(-1.5756109914268008) q[4];
rz(0.8460242865212378) q[4];
ry(2.0232896216181304) q[5];
rz(-1.1154089337552238) q[5];
ry(2.0660316752325336) q[6];
rz(1.5344850223739037) q[6];
ry(-0.4767345096566588) q[7];
rz(-0.2723047748933575) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.46551004030966686) q[0];
rz(1.760888783197972) q[0];
ry(-1.6481710921548425) q[1];
rz(0.03860560247506787) q[1];
ry(2.732779958691255) q[2];
rz(1.6706086447053377) q[2];
ry(1.0235782428426061) q[3];
rz(-2.697261973250571) q[3];
ry(-0.9059682355645621) q[4];
rz(-0.11078505947593432) q[4];
ry(0.6347765745927525) q[5];
rz(2.664335765736401) q[5];
ry(2.3072668801506535) q[6];
rz(1.040624548052091) q[6];
ry(0.916376503337232) q[7];
rz(-2.0686628928119446) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.8479236694236758) q[0];
rz(-2.584955911572513) q[0];
ry(0.4900398001656612) q[1];
rz(-2.387817985775625) q[1];
ry(2.276532314944829) q[2];
rz(-0.9967065754985133) q[2];
ry(-0.04979037532828645) q[3];
rz(0.5102333103025867) q[3];
ry(-0.26438561061705024) q[4];
rz(-2.288666708745848) q[4];
ry(2.5328542185735397) q[5];
rz(0.27094576498012324) q[5];
ry(1.2458601000394056) q[6];
rz(0.2928923340452636) q[6];
ry(0.2988628959804824) q[7];
rz(1.2085612743914447) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.5235560519526414) q[0];
rz(-0.4937744040402195) q[0];
ry(-1.4927356057419103) q[1];
rz(-2.798184355023396) q[1];
ry(2.7981285740719146) q[2];
rz(0.7192193138644551) q[2];
ry(1.9052875776382123) q[3];
rz(1.5268660243782923) q[3];
ry(0.11837485760277922) q[4];
rz(-1.6257323283547584) q[4];
ry(-1.0237849075868335) q[5];
rz(1.5329191348304052) q[5];
ry(-1.1441322654317592) q[6];
rz(-1.211714198600177) q[6];
ry(-2.9869303015466704) q[7];
rz(0.4294595113832002) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.301826501791236) q[0];
rz(-2.068185162480005) q[0];
ry(0.6601112638509249) q[1];
rz(-1.3589293926004016) q[1];
ry(-0.39219729789327573) q[2];
rz(0.572703009185905) q[2];
ry(2.0740984459877225) q[3];
rz(-1.4642908324341217) q[3];
ry(0.13044885096929448) q[4];
rz(-1.2774015443868807) q[4];
ry(-0.7881651982183513) q[5];
rz(0.3776933890468559) q[5];
ry(-1.7645376710669423) q[6];
rz(-0.8424500152940314) q[6];
ry(2.9158065663235755) q[7];
rz(2.3975157294967726) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.83294758245009) q[0];
rz(-1.3149222090305634) q[0];
ry(2.9581901518924623) q[1];
rz(1.7953292576806772) q[1];
ry(1.1340875126751504) q[2];
rz(2.387346974307988) q[2];
ry(0.667328035739204) q[3];
rz(2.6229778080974704) q[3];
ry(-2.6754209386781365) q[4];
rz(-3.1016064286013507) q[4];
ry(-1.9708702334546908) q[5];
rz(-0.5974225972161852) q[5];
ry(0.50500804273649) q[6];
rz(-0.714017307406721) q[6];
ry(-1.1966001224310867) q[7];
rz(2.867917252542295) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.332000694862464) q[0];
rz(0.9756496778164435) q[0];
ry(-2.714590322762332) q[1];
rz(-1.5991833512675706) q[1];
ry(-2.516350721547028) q[2];
rz(2.5486124584352425) q[2];
ry(-0.32569744366434317) q[3];
rz(2.3245044476495913) q[3];
ry(-2.4900394546793185) q[4];
rz(-2.4420357213947783) q[4];
ry(1.0664397646515802) q[5];
rz(0.9855552459961495) q[5];
ry(-2.567634719667689) q[6];
rz(-2.546855866363748) q[6];
ry(0.47899872658677195) q[7];
rz(2.5765072813464394) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.0028176707660012563) q[0];
rz(1.7242224914472182) q[0];
ry(-2.911483095944632) q[1];
rz(0.14506306800119934) q[1];
ry(1.825586180888565) q[2];
rz(0.4918628157651595) q[2];
ry(1.8690866790553138) q[3];
rz(-3.126598647982462) q[3];
ry(2.9570173634788226) q[4];
rz(0.2708732089855164) q[4];
ry(-1.8078190979388395) q[5];
rz(2.6530801285441434) q[5];
ry(1.3737979956696653) q[6];
rz(0.989433272111323) q[6];
ry(2.881083889198346) q[7];
rz(-1.15103994480332) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.3161630017151595) q[0];
rz(1.462419339762592) q[0];
ry(1.4818102215454187) q[1];
rz(2.761156096360271) q[1];
ry(-0.44730925898381474) q[2];
rz(-3.0328571144952767) q[2];
ry(-1.6654257266389936) q[3];
rz(-0.0016587302499687633) q[3];
ry(-0.4021027020788145) q[4];
rz(-1.2637897516096022) q[4];
ry(-1.0390680717458012) q[5];
rz(-0.5041081930601878) q[5];
ry(0.7983289667867508) q[6];
rz(-2.8714106674022863) q[6];
ry(-1.53821026809897) q[7];
rz(-0.06249416222923454) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.017877878418969928) q[0];
rz(0.845883056301096) q[0];
ry(0.8383575798453009) q[1];
rz(1.3383376842026986) q[1];
ry(-2.6483101481706925) q[2];
rz(2.263677885464727) q[2];
ry(2.995005101310704) q[3];
rz(2.2148989013534273) q[3];
ry(-0.902258123892005) q[4];
rz(2.9423436452862117) q[4];
ry(2.7694012883026073) q[5];
rz(-0.7479948495089367) q[5];
ry(-2.32054621176415) q[6];
rz(-2.1866329091104615) q[6];
ry(-2.0998029728629857) q[7];
rz(0.4409321956808414) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.9991840809579573) q[0];
rz(2.2141143315098253) q[0];
ry(0.1778629563031641) q[1];
rz(2.197470371918446) q[1];
ry(-0.3021299515923701) q[2];
rz(1.502526505383982) q[2];
ry(-3.0374880007542036) q[3];
rz(-1.2541342495635095) q[3];
ry(-1.6841652517319758) q[4];
rz(-2.148212832847107) q[4];
ry(0.7522821744419823) q[5];
rz(1.389242332279337) q[5];
ry(-1.1173884849963631) q[6];
rz(2.729066437734153) q[6];
ry(-1.180150572902619) q[7];
rz(2.7766976263778362) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.324237076202494) q[0];
rz(-2.313022252969557) q[0];
ry(-0.7481315232329533) q[1];
rz(0.7494251269400863) q[1];
ry(0.6842590856202871) q[2];
rz(1.0689622730420059) q[2];
ry(3.0360483473851687) q[3];
rz(0.7892807654086802) q[3];
ry(-2.775371986659276) q[4];
rz(-0.6940869717474394) q[4];
ry(0.6940849500797128) q[5];
rz(-0.28093219275944037) q[5];
ry(2.5770321269375915) q[6];
rz(-1.0660763333197814) q[6];
ry(2.645666249074519) q[7];
rz(1.1369913051777427) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.793063094906966) q[0];
rz(2.361050076761208) q[0];
ry(0.9396314731774433) q[1];
rz(2.6384632134082455) q[1];
ry(-1.6392049423708497) q[2];
rz(2.1399838755829146) q[2];
ry(-1.362641675816617) q[3];
rz(1.0724747046779042) q[3];
ry(0.5448094761850545) q[4];
rz(0.5760691330307239) q[4];
ry(1.5678839090182441) q[5];
rz(0.9395706213240498) q[5];
ry(0.31630804079263086) q[6];
rz(2.505353863709301) q[6];
ry(-2.320409076166176) q[7];
rz(-1.085427367198152) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.426756052620089) q[0];
rz(-0.3836867442362833) q[0];
ry(1.2727397852275644) q[1];
rz(-0.5004441211091342) q[1];
ry(-0.9757983321998586) q[2];
rz(0.14454219744648003) q[2];
ry(2.595350936163126) q[3];
rz(-2.3430986291988494) q[3];
ry(1.9960872008974677) q[4];
rz(-3.1044559192824566) q[4];
ry(1.593517941276974) q[5];
rz(-1.8018008011859008) q[5];
ry(-1.5688257135805141) q[6];
rz(-2.583579378017993) q[6];
ry(1.5273513476890088) q[7];
rz(-2.8983349828916425) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.0817125012213957) q[0];
rz(2.9236405324794066) q[0];
ry(-1.726350595267779) q[1];
rz(-3.032736062138111) q[1];
ry(-3.027167039344981) q[2];
rz(1.297803218123116) q[2];
ry(0.7072719748657209) q[3];
rz(1.0343404238292009) q[3];
ry(-2.6664240275839575) q[4];
rz(0.2847482434556895) q[4];
ry(2.578922242518864) q[5];
rz(2.573275537298191) q[5];
ry(0.6181594915193322) q[6];
rz(1.1456954611489447) q[6];
ry(2.5485714279232634) q[7];
rz(-1.382990834687104) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.009121050857576307) q[0];
rz(-0.9512242150199207) q[0];
ry(2.2014120672167934) q[1];
rz(2.704729023941913) q[1];
ry(0.2745299403443022) q[2];
rz(-2.731569682271972) q[2];
ry(-1.3128274701108555) q[3];
rz(2.510881954308193) q[3];
ry(1.4016820708461344) q[4];
rz(2.4935829325461896) q[4];
ry(-1.3043515789627806) q[5];
rz(-2.6827410656147213) q[5];
ry(-0.8828307986245774) q[6];
rz(2.6224802821603) q[6];
ry(2.351087553566469) q[7];
rz(-1.6388672784167797) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.550861829977764) q[0];
rz(1.12286962672402) q[0];
ry(1.6245220525676152) q[1];
rz(2.0043881919839683) q[1];
ry(-2.0206507683110893) q[2];
rz(2.9768246823365936) q[2];
ry(-2.8272395508073265) q[3];
rz(-0.7906043652836052) q[3];
ry(-2.641992948766814) q[4];
rz(2.8016440532673554) q[4];
ry(0.2885760000870032) q[5];
rz(-2.5129164709905223) q[5];
ry(-0.8804097742016435) q[6];
rz(-0.6254077000681459) q[6];
ry(0.7871225764438338) q[7];
rz(0.17390598655173317) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.2460312314254685) q[0];
rz(-0.16995282334749717) q[0];
ry(0.5776356273757403) q[1];
rz(-2.578749759310684) q[1];
ry(-1.2625517306645393) q[2];
rz(-1.6703936081779427) q[2];
ry(-0.7606105760237363) q[3];
rz(-1.9912675598779732) q[3];
ry(2.384277092045135) q[4];
rz(0.027189117544859087) q[4];
ry(-2.8136966635462617) q[5];
rz(0.9489757260185933) q[5];
ry(-1.2017465820723006) q[6];
rz(0.7906790743726544) q[6];
ry(-0.2545633174941111) q[7];
rz(0.7040243461779409) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.6377192719475133) q[0];
rz(0.1263957369013214) q[0];
ry(2.2099751595958117) q[1];
rz(-0.04446567828490089) q[1];
ry(-0.823344241133416) q[2];
rz(0.9577928846467209) q[2];
ry(-1.8483542186135908) q[3];
rz(2.181636230309123) q[3];
ry(-0.7919187668390864) q[4];
rz(0.16815662377348042) q[4];
ry(0.7697905366300332) q[5];
rz(-2.346352524809277) q[5];
ry(0.728556183395537) q[6];
rz(2.895133427262844) q[6];
ry(2.886998093352288) q[7];
rz(-1.044294067782525) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.2258444200462864) q[0];
rz(1.6565003637522056) q[0];
ry(-2.3399642283174233) q[1];
rz(0.923351477210792) q[1];
ry(1.7483506561827402) q[2];
rz(-0.4689417285559241) q[2];
ry(-2.2521831810855826) q[3];
rz(-1.0666557130621168) q[3];
ry(-1.4675842032164716) q[4];
rz(0.4310742975432421) q[4];
ry(-1.6373784738353976) q[5];
rz(1.3653266772871522) q[5];
ry(2.587114522092302) q[6];
rz(-1.2919591573886153) q[6];
ry(-1.6215115090104035) q[7];
rz(0.7751772021877468) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.20209515303374215) q[0];
rz(-2.0476631167948036) q[0];
ry(1.4695909715378432) q[1];
rz(1.5903664390096557) q[1];
ry(2.734441193343514) q[2];
rz(-1.2324534026226113) q[2];
ry(0.6661687711465404) q[3];
rz(0.16857486347687523) q[3];
ry(-1.8202411126312046) q[4];
rz(0.6362191499536892) q[4];
ry(-0.17073103279759835) q[5];
rz(-1.008703656129411) q[5];
ry(0.21792196390384136) q[6];
rz(-2.394729281193003) q[6];
ry(1.3717615414584738) q[7];
rz(-2.0715075839380632) q[7];