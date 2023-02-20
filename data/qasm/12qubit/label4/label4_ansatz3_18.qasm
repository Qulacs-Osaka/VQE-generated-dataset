OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.9270463247538943) q[0];
rz(2.1726332872509566) q[0];
ry(-1.7216655422712277) q[1];
rz(-1.519825556204056) q[1];
ry(-3.0778524443433914) q[2];
rz(-0.3511598639186868) q[2];
ry(-1.0495452815467985) q[3];
rz(1.9621361837855666) q[3];
ry(-0.07388010817766748) q[4];
rz(1.0934437835173825) q[4];
ry(-3.138541110400773) q[5];
rz(-2.839658242045002) q[5];
ry(2.9659621942126217) q[6];
rz(1.7831221185825783) q[6];
ry(-1.1723925476490982) q[7];
rz(1.0813669458189497) q[7];
ry(-0.27901501323486055) q[8];
rz(-0.03093510032025737) q[8];
ry(-2.198384007664809) q[9];
rz(-2.123365897686411) q[9];
ry(-2.8938249532705433) q[10];
rz(-1.4156995851200982) q[10];
ry(-2.077943332507882) q[11];
rz(1.493934685212387) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.8279932915618) q[0];
rz(1.1114346261535113) q[0];
ry(-0.06430347421587551) q[1];
rz(-0.7080779599176718) q[1];
ry(0.043982175855654226) q[2];
rz(-2.450880828268128) q[2];
ry(-1.5145257794978602) q[3];
rz(2.62515988979335) q[3];
ry(-0.23617520364891573) q[4];
rz(1.817273730264426) q[4];
ry(0.000450068843511253) q[5];
rz(-2.953265196439415) q[5];
ry(-3.0648209722031687) q[6];
rz(-0.9993482878871787) q[6];
ry(-2.3991213080998888) q[7];
rz(1.5524861090567912) q[7];
ry(0.6848068866499162) q[8];
rz(-1.7860938054468216) q[8];
ry(-2.1937899232747835) q[9];
rz(-0.9673038007271182) q[9];
ry(-1.0820146537319904) q[10];
rz(-2.1103030232499362) q[10];
ry(2.0873165977193513) q[11];
rz(0.35569036840749924) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.797626213101825) q[0];
rz(2.7336064416523103) q[0];
ry(0.1439960628199497) q[1];
rz(0.7224907613246833) q[1];
ry(0.2736831052422328) q[2];
rz(-3.044216130014287) q[2];
ry(-1.5170496852836366) q[3];
rz(1.608823339108767) q[3];
ry(-0.0491360566303042) q[4];
rz(-0.5079338842290301) q[4];
ry(0.4226601763602463) q[5];
rz(2.8393851375450954) q[5];
ry(0.008736702984761124) q[6];
rz(-2.3280564209231196) q[6];
ry(-3.1246225018636715) q[7];
rz(0.8569972220580669) q[7];
ry(-1.4399340348127252) q[8];
rz(0.9975600912826037) q[8];
ry(-2.5595117201879476) q[9];
rz(1.4268156232731126) q[9];
ry(2.100781252544879) q[10];
rz(-3.0515483584893253) q[10];
ry(1.2907564227096466) q[11];
rz(-0.08876755591904839) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.2122057998468159) q[0];
rz(1.8404332785728625) q[0];
ry(-1.1811355196092324) q[1];
rz(2.2541169282216402) q[1];
ry(-0.030130504640032953) q[2];
rz(1.2577880685404175) q[2];
ry(0.2305959003328948) q[3];
rz(-1.5630076321163315) q[3];
ry(0.018433059813098396) q[4];
rz(-2.4242002353645953) q[4];
ry(-3.1410563760466674) q[5];
rz(3.026140031788395) q[5];
ry(-3.1316010133197167) q[6];
rz(-0.5905372182148866) q[6];
ry(3.1391693877019407) q[7];
rz(2.2432035751306625) q[7];
ry(2.249597612479949) q[8];
rz(-0.5961767793546677) q[8];
ry(1.4257437529584882) q[9];
rz(-1.4087662987094234) q[9];
ry(0.1935449970814984) q[10];
rz(2.9945266967581814) q[10];
ry(-2.3523375465414498) q[11];
rz(2.418254659471571) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.0115696837505226) q[0];
rz(2.9896272294901496) q[0];
ry(1.3516853710956054) q[1];
rz(-2.7406373791023233) q[1];
ry(2.0285716758479246) q[2];
rz(-0.6277239324557113) q[2];
ry(2.3020872627852524) q[3];
rz(-0.3962747320231968) q[3];
ry(2.27875201338915) q[4];
rz(2.9543584131230536) q[4];
ry(-2.710715663311864) q[5];
rz(0.07614202746104233) q[5];
ry(3.0873418457625315) q[6];
rz(2.4721587083497094) q[6];
ry(-1.707991954913349) q[7];
rz(2.7387698826808626) q[7];
ry(-1.3954106620459668) q[8];
rz(1.4795558354606004) q[8];
ry(1.5220187276391766) q[9];
rz(-2.818019023221036) q[9];
ry(-1.3170952223098409) q[10];
rz(2.7036953548551943) q[10];
ry(2.442184143034658) q[11];
rz(2.359450437659864) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.7808079372041927) q[0];
rz(2.8324950635096062) q[0];
ry(-2.187519229035191) q[1];
rz(0.5378021149543971) q[1];
ry(0.09241778983221503) q[2];
rz(-0.4911726010910536) q[2];
ry(-0.43610873345141465) q[3];
rz(-1.920102151905568) q[3];
ry(-0.002101217668040611) q[4];
rz(-2.9543528386051072) q[4];
ry(3.1405809369192483) q[5];
rz(-1.6157648476978468) q[5];
ry(-0.002155596240839941) q[6];
rz(2.4608331394146843) q[6];
ry(0.0040077433707912296) q[7];
rz(0.17766597999595793) q[7];
ry(0.9460852687561464) q[8];
rz(1.0181355707103215) q[8];
ry(0.7408666024965322) q[9];
rz(-0.3072484543789349) q[9];
ry(1.510116391647316) q[10];
rz(-2.275825430710496) q[10];
ry(1.2355990609050491) q[11];
rz(-2.640419088645675) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.703612043514801) q[0];
rz(2.537178894317641) q[0];
ry(-2.9380796209371014) q[1];
rz(-2.3616038910863266) q[1];
ry(-2.666612372925636) q[2];
rz(-1.576618036495713) q[2];
ry(-1.63184683846291) q[3];
rz(2.876283314433404) q[3];
ry(2.3935726756869857) q[4];
rz(2.416889848351802) q[4];
ry(1.6130361242188922) q[5];
rz(0.09730902531267195) q[5];
ry(3.084046316584557) q[6];
rz(-0.4866601934864428) q[6];
ry(-1.3705445222090704) q[7];
rz(-2.571644282797092) q[7];
ry(-1.6729236541759738) q[8];
rz(-1.7715945172192775) q[8];
ry(2.444091909349637) q[9];
rz(1.8026387927361829) q[9];
ry(-2.5780018194083656) q[10];
rz(1.9633104001360513) q[10];
ry(-1.5765991581873549) q[11];
rz(-1.7907762779713456) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.0150074678300935) q[0];
rz(-1.5901827025192148) q[0];
ry(-1.8019821561293445) q[1];
rz(2.9624750261757713) q[1];
ry(-3.109257214967301) q[2];
rz(-0.8065648206055532) q[2];
ry(-1.581986788104261) q[3];
rz(-2.7346635826637895) q[3];
ry(-3.1113800838099857) q[4];
rz(3.1368935942829697) q[4];
ry(-3.1415469565030008) q[5];
rz(-1.5380710542808087) q[5];
ry(-2.172733335050868) q[6];
rz(-0.27351699656046685) q[6];
ry(6.184997056646324e-05) q[7];
rz(-1.194962509004978) q[7];
ry(-0.5923604508283384) q[8];
rz(-0.032645896950332265) q[8];
ry(0.5124405534995429) q[9];
rz(-2.1526755889337204) q[9];
ry(0.5045064039183872) q[10];
rz(2.079321910765459) q[10];
ry(1.5374020685138774) q[11];
rz(-0.5556617289259833) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.977678951745501) q[0];
rz(1.0346316770329904) q[0];
ry(-1.5678201005880146) q[1];
rz(-0.35339055025183264) q[1];
ry(-1.7329950529325515) q[2];
rz(-1.5812203366579256) q[2];
ry(3.0501692348183145) q[3];
rz(-0.8520854452630257) q[3];
ry(-3.1170765792703423) q[4];
rz(1.808649701529329) q[4];
ry(-0.44439662832175586) q[5];
rz(0.6552298030165672) q[5];
ry(3.0548856807238294) q[6];
rz(-2.2256977038375707) q[6];
ry(2.6699554578564118) q[7];
rz(-0.17187433411689135) q[7];
ry(0.2695731147639239) q[8];
rz(1.0383771456704718) q[8];
ry(-2.308229700570485) q[9];
rz(0.3268288318192214) q[9];
ry(1.4204853206078778) q[10];
rz(-0.25646205508409287) q[10];
ry(2.439834779639059) q[11];
rz(0.22419823629763147) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.8083031605451128) q[0];
rz(2.4573984454889843) q[0];
ry(3.106808199550357) q[1];
rz(1.2031203755312943) q[1];
ry(-2.9909624249585143) q[2];
rz(-1.788677495635711) q[2];
ry(-0.20516493340016417) q[3];
rz(3.1276589404626884) q[3];
ry(-2.9231173000135575) q[4];
rz(0.9832702417694728) q[4];
ry(3.141131717447139) q[5];
rz(-2.5878442599210065) q[5];
ry(2.027559862002354) q[6];
rz(-2.7949066194163352) q[6];
ry(0.000711492558357752) q[7];
rz(2.9882932601209378) q[7];
ry(1.6394073864446153) q[8];
rz(-3.0104244560216) q[8];
ry(1.6330977851215214) q[9];
rz(1.184845423750251) q[9];
ry(-2.764647970695832) q[10];
rz(1.2354825536940726) q[10];
ry(2.540534355526385) q[11];
rz(-0.4800421153991636) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.602360156095834) q[0];
rz(2.679336036559164) q[0];
ry(-0.3824145173450937) q[1];
rz(0.987044808784222) q[1];
ry(-3.050563000694763) q[2];
rz(2.768776716499592) q[2];
ry(1.3549234265129417) q[3];
rz(-2.8639400130277783) q[3];
ry(0.020689041736591918) q[4];
rz(-1.4003423963260815) q[4];
ry(-2.216710929047702) q[5];
rz(1.6453940342840792) q[5];
ry(-3.141160043445158) q[6];
rz(2.538710644362409) q[6];
ry(2.0908435029464005) q[7];
rz(-1.0580417877577837) q[7];
ry(-1.583471310279748) q[8];
rz(-1.7145187466509721) q[8];
ry(-1.7764338089389347) q[9];
rz(2.230104828183917) q[9];
ry(1.2406937538842868) q[10];
rz(1.5515882924035895) q[10];
ry(0.7810654166374116) q[11];
rz(2.5064920381129414) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.3461855083935768) q[0];
rz(2.422612306879451) q[0];
ry(-0.009910188784502557) q[1];
rz(-2.690855138345519) q[1];
ry(-3.08046609356452) q[2];
rz(0.2706132637919043) q[2];
ry(-2.4185983306086047) q[3];
rz(-3.034790652619629) q[3];
ry(0.003687138692764158) q[4];
rz(1.9168838059823257) q[4];
ry(-3.1410248200228903) q[5];
rz(-1.2420646631429597) q[5];
ry(0.7440980402676136) q[6];
rz(1.0541260303185762) q[6];
ry(-0.0008929244704489747) q[7];
rz(-2.570539998308015) q[7];
ry(-2.9104191827233548) q[8];
rz(-1.3469631580537964) q[8];
ry(1.5817470019613813) q[9];
rz(-2.582615553580237) q[9];
ry(1.6714542583621572) q[10];
rz(-1.736258068024783) q[10];
ry(-1.74948213080702) q[11];
rz(2.6343014460476932) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.0563268399342358) q[0];
rz(0.4236387257477663) q[0];
ry(-0.0057553231653013626) q[1];
rz(-0.6154504509290682) q[1];
ry(2.551052761216006) q[2];
rz(0.8613552083239409) q[2];
ry(-1.3892435010323474) q[3];
rz(0.17496046462143874) q[3];
ry(1.5664953849994714) q[4];
rz(0.7875204382365676) q[4];
ry(-2.1978461991970297) q[5];
rz(1.6937166632074647) q[5];
ry(-3.1415134676825454) q[6];
rz(1.7406300392948828) q[6];
ry(0.7498787644733946) q[7];
rz(1.3236849638534949) q[7];
ry(-3.0739162973452943) q[8];
rz(-1.9185359618910738) q[8];
ry(-0.4968677797821348) q[9];
rz(-0.42871224524310847) q[9];
ry(0.8942837022356621) q[10];
rz(-2.6337536689033136) q[10];
ry(-2.9852219871853904) q[11];
rz(0.14971728476084403) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.0616589597988724) q[0];
rz(-1.2078562685797025) q[0];
ry(2.538888841357516) q[1];
rz(2.279717036064153) q[1];
ry(1.5706324401949896) q[2];
rz(1.5699864288912928) q[2];
ry(-2.391911711506341) q[3];
rz(2.647414534909422) q[3];
ry(3.1404440275660406) q[4];
rz(-2.3328169905062524) q[4];
ry(-0.03481813623010833) q[5];
rz(0.18833038742255498) q[5];
ry(-1.1285251387603497) q[6];
rz(-1.1539741705699347) q[6];
ry(-3.1409664177781353) q[7];
rz(0.1173146121437547) q[7];
ry(-1.5745968515694218) q[8];
rz(1.4147404947661784) q[8];
ry(2.3321814739107367) q[9];
rz(-2.742661715076884) q[9];
ry(-0.4230581890773464) q[10];
rz(0.025465436132076746) q[10];
ry(0.47966543140160534) q[11];
rz(1.0660086796891235) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.008749398514069742) q[0];
rz(2.3659474223758106) q[0];
ry(0.0019899429014218484) q[1];
rz(-3.09785149849688) q[1];
ry(1.5740055500209955) q[2];
rz(-1.9916805931649915) q[2];
ry(-3.140852931283113) q[3];
rz(-0.9926029720614062) q[3];
ry(0.0427909059646776) q[4];
rz(-1.594511416014943) q[4];
ry(0.23559192001126225) q[5];
rz(-1.6943605827519548) q[5];
ry(-3.137649731548078) q[6];
rz(-1.144803582960515) q[6];
ry(-0.0021767288882301042) q[7];
rz(0.850421376343382) q[7];
ry(3.136760268319987) q[8];
rz(2.28477689815369) q[8];
ry(0.028144991637741975) q[9];
rz(2.6429812124090994) q[9];
ry(-0.949550235611306) q[10];
rz(-0.08299432844743038) q[10];
ry(-1.2672321495955359) q[11];
rz(1.8833060850975933) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(8.311645701135414e-05) q[0];
rz(0.5304492213107954) q[0];
ry(-1.8852933669160319) q[1];
rz(0.5680199017229874) q[1];
ry(-0.2981931017015471) q[2];
rz(-2.7650537332727345) q[2];
ry(-1.5701775523525683) q[3];
rz(-2.389319519427371) q[3];
ry(-3.1299055935712548) q[4];
rz(-0.7822398678494691) q[4];
ry(1.5683689301367147) q[5];
rz(1.6058866740154298) q[5];
ry(1.3750039781798347) q[6];
rz(-2.18809180255624) q[6];
ry(0.0008561205541948169) q[7];
rz(-2.244522518504235) q[7];
ry(0.22190997323805825) q[8];
rz(0.6322443645087477) q[8];
ry(0.05967380285174409) q[9];
rz(3.0089868339631596) q[9];
ry(-1.049424858949906) q[10];
rz(0.14288537795535916) q[10];
ry(-2.147238727919879) q[11];
rz(1.8009772049141077) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5731801500199891) q[0];
rz(1.5707255384265326) q[0];
ry(-3.141026436715305) q[1];
rz(2.365136655966687) q[1];
ry(-2.9252538359098934) q[2];
rz(-1.15115542809838) q[2];
ry(-0.0003889336256053255) q[3];
rz(-2.7327828641429384) q[3];
ry(0.0005697114461366852) q[4];
rz(0.7790089676878801) q[4];
ry(-1.5709204814722377) q[5];
rz(-1.7848313651797731) q[5];
ry(3.135518006741434) q[6];
rz(-2.2285896789288735) q[6];
ry(0.00033390842515845884) q[7];
rz(2.770312183824054) q[7];
ry(-2.3933706297232344) q[8];
rz(-1.5708922024288867) q[8];
ry(2.5284773458608782) q[9];
rz(-1.5492463851864444) q[9];
ry(2.071127295630638) q[10];
rz(-3.1111784633100608) q[10];
ry(1.9868362911225486) q[11];
rz(0.7333874434392431) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.7467515198289556) q[0];
rz(-1.2249780844224505) q[0];
ry(1.5342351904548865) q[1];
rz(0.473323300113373) q[1];
ry(1.5706779897617815) q[2];
rz(1.5756813750490997) q[2];
ry(3.1415612618809785) q[3];
rz(2.725607850299748) q[3];
ry(-1.5726597432287395) q[4];
rz(2.043340222392701) q[4];
ry(-0.0005608609846122037) q[5];
rz(2.4697601573872205) q[5];
ry(2.6417763754036234) q[6];
rz(-2.014400167917237) q[6];
ry(-2.711289333591699) q[7];
rz(1.16229707012974) q[7];
ry(-1.5671826606454413) q[8];
rz(0.15361637682954132) q[8];
ry(1.5594646128209133) q[9];
rz(-1.577795444790823) q[9];
ry(-3.1139923893329096) q[10];
rz(-2.9249478026752986) q[10];
ry(0.9033795892247029) q[11];
rz(-0.829590459874665) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.00190590224563536) q[0];
rz(2.3760803980351937) q[0];
ry(1.1597742943101077) q[1];
rz(3.1076632635946546) q[1];
ry(-1.571914376578682) q[2];
rz(1.7988129775606791) q[2];
ry(-1.5315846122937424) q[3];
rz(-3.0145585815089784) q[3];
ry(0.0004477678129388707) q[4];
rz(2.572084744056175) q[4];
ry(-6.575128181918188e-05) q[5];
rz(2.4162872518113767) q[5];
ry(-0.0013521797547126023) q[6];
rz(2.0067273399429375) q[6];
ry(3.140626663617156) q[7];
rz(-0.8143575323554034) q[7];
ry(-1.6039518669054404) q[8];
rz(-0.10679874066979789) q[8];
ry(1.186915212109738) q[9];
rz(-1.494304521503155) q[9];
ry(-1.6229822557607356) q[10];
rz(3.122853662862038) q[10];
ry(5.0809160944662024e-05) q[11];
rz(-1.742897069388361) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.1410545810184907) q[0];
rz(1.5920013614462976) q[0];
ry(2.413697009053364) q[1];
rz(0.0009396964234959755) q[1];
ry(3.139810149982155) q[2];
rz(0.9209093163859747) q[2];
ry(3.131755569367115) q[3];
rz(3.1292523878487857) q[3];
ry(0.1736698177830241) q[4];
rz(-0.0908239553583492) q[4];
ry(-0.0009194876766782655) q[5];
rz(-3.0617676066463133) q[5];
ry(1.4654321850034622) q[6];
rz(2.390042662922509) q[6];
ry(-2.6781729286261085) q[7];
rz(-1.666173316309904) q[7];
ry(-0.001178949806345919) q[8];
rz(0.05784305450047894) q[8];
ry(-3.1366553110987203) q[9];
rz(0.06969268089712888) q[9];
ry(-2.3057243465466137) q[10];
rz(-1.6246232023776324) q[10];
ry(-0.008840144329522737) q[11];
rz(-0.08421506832959781) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.0009311480399282143) q[0];
rz(-0.4419444352789311) q[0];
ry(-1.9987625450102033) q[1];
rz(-1.9213998034057331) q[1];
ry(-3.1356381601581726) q[2];
rz(-2.61587137197244) q[2];
ry(-0.05821096189475484) q[3];
rz(-2.818179302046185) q[3];
ry(0.00038660588478754763) q[4];
rz(-1.3840212387832738) q[4];
ry(3.1394997171109553) q[5];
rz(0.038682131496657834) q[5];
ry(0.0015742543029411138) q[6];
rz(-2.8832661409636318) q[6];
ry(-3.0688122384439973) q[7];
rz(1.1214250474724539) q[7];
ry(0.22865979717016233) q[8];
rz(-0.8061298054637447) q[8];
ry(-1.5586216471117709) q[9];
rz(1.2079964923174726) q[9];
ry(1.0242024304799422) q[10];
rz(-3.0830126050717555) q[10];
ry(0.0046881107929124255) q[11];
rz(-2.5429622457547243) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.5701683510713489) q[0];
rz(1.4827577196342236) q[0];
ry(1.273332224370109) q[1];
rz(1.4117003677057838) q[1];
ry(-3.139271977112671) q[2];
rz(-1.828431634071653) q[2];
ry(-0.003865278555137408) q[3];
rz(-1.2397099894674577) q[3];
ry(-1.5883983807020432) q[4];
rz(2.8926309574607583) q[4];
ry(1.5727692887276954) q[5];
rz(2.087128502640331) q[5];
ry(-0.11294836193939997) q[6];
rz(-2.7291894290669156) q[6];
ry(-0.005734528321275462) q[7];
rz(0.9630898874854248) q[7];
ry(3.141056620333892) q[8];
rz(2.2224059511130774) q[8];
ry(3.1376299678490986) q[9];
rz(-1.7405913830361213) q[9];
ry(-1.5651885307280897) q[10];
rz(3.0149719524711016) q[10];
ry(-3.140013881212991) q[11];
rz(1.523640361327967) q[11];