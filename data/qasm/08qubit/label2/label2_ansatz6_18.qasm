OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.9348760571911692) q[0];
ry(-3.1012401365658313) q[1];
cx q[0],q[1];
ry(0.40669931335373677) q[0];
ry(2.779800154426104) q[1];
cx q[0],q[1];
ry(-1.9332739758640551) q[1];
ry(2.409895582553449) q[2];
cx q[1],q[2];
ry(0.9147283169153816) q[1];
ry(-2.4314906908200498) q[2];
cx q[1],q[2];
ry(-0.30084587156328707) q[2];
ry(0.7471821332640622) q[3];
cx q[2],q[3];
ry(-0.1453336023096785) q[2];
ry(2.931725499069928) q[3];
cx q[2],q[3];
ry(-0.8683409460872041) q[3];
ry(1.244180347397895) q[4];
cx q[3],q[4];
ry(1.1837636633584339) q[3];
ry(-2.370665486180486) q[4];
cx q[3],q[4];
ry(2.92772126965124) q[4];
ry(1.4616996008336223) q[5];
cx q[4],q[5];
ry(-2.248981676517676) q[4];
ry(-0.40125834755590795) q[5];
cx q[4],q[5];
ry(-0.20914482449168317) q[5];
ry(-0.033059099853042895) q[6];
cx q[5],q[6];
ry(-1.728096767794561) q[5];
ry(-0.20286166980584025) q[6];
cx q[5],q[6];
ry(0.545856660231217) q[6];
ry(1.3362114246505925) q[7];
cx q[6],q[7];
ry(2.196079198039789) q[6];
ry(3.025660109887669) q[7];
cx q[6],q[7];
ry(1.3955029399650272) q[0];
ry(-2.1490386986709167) q[1];
cx q[0],q[1];
ry(2.783429202602018) q[0];
ry(2.229180305706393) q[1];
cx q[0],q[1];
ry(-2.513335958024293) q[1];
ry(0.12397482587992494) q[2];
cx q[1],q[2];
ry(-1.423152333105521) q[1];
ry(0.27816236385615944) q[2];
cx q[1],q[2];
ry(-1.7027009047385415) q[2];
ry(-0.882963038244622) q[3];
cx q[2],q[3];
ry(2.7961785216758295) q[2];
ry(-2.5781346470351676) q[3];
cx q[2],q[3];
ry(-0.4200353384704255) q[3];
ry(-1.431373805880706) q[4];
cx q[3],q[4];
ry(-3.0155546741083135) q[3];
ry(0.7529903915134691) q[4];
cx q[3],q[4];
ry(2.076732562881421) q[4];
ry(-3.0678011223244814) q[5];
cx q[4],q[5];
ry(0.32370776271078044) q[4];
ry(-2.4097428422488125) q[5];
cx q[4],q[5];
ry(1.6245686075634096) q[5];
ry(-0.14404321268134004) q[6];
cx q[5],q[6];
ry(1.0379159211275546) q[5];
ry(2.597817906961419) q[6];
cx q[5],q[6];
ry(-2.9811415936458308) q[6];
ry(0.2957756914004376) q[7];
cx q[6],q[7];
ry(-0.6716436326577353) q[6];
ry(-1.1477722759406328) q[7];
cx q[6],q[7];
ry(1.7093956020910968) q[0];
ry(-2.194432974524262) q[1];
cx q[0],q[1];
ry(-2.1613859968657465) q[0];
ry(0.13157287836495352) q[1];
cx q[0],q[1];
ry(-1.7192589858120522) q[1];
ry(-0.37897992708047834) q[2];
cx q[1],q[2];
ry(-2.714404383625613) q[1];
ry(-1.7229349747299871) q[2];
cx q[1],q[2];
ry(-3.124958438099698) q[2];
ry(0.9705512193621715) q[3];
cx q[2],q[3];
ry(-2.359499612659231) q[2];
ry(-0.22510462523303598) q[3];
cx q[2],q[3];
ry(-0.8256668467111109) q[3];
ry(-0.9188414830546279) q[4];
cx q[3],q[4];
ry(-2.5409841651511926) q[3];
ry(0.9988762790008389) q[4];
cx q[3],q[4];
ry(2.587045166421062) q[4];
ry(0.7391218651607557) q[5];
cx q[4],q[5];
ry(-1.2260162043833411) q[4];
ry(-0.7271844078797352) q[5];
cx q[4],q[5];
ry(-0.21693415979937303) q[5];
ry(-0.005921977962058727) q[6];
cx q[5],q[6];
ry(2.8156011319053795) q[5];
ry(1.3491846306259143) q[6];
cx q[5],q[6];
ry(-0.3396134274461768) q[6];
ry(-2.772772191293502) q[7];
cx q[6],q[7];
ry(2.372368924099343) q[6];
ry(1.4104977024644878) q[7];
cx q[6],q[7];
ry(1.240751109277494) q[0];
ry(2.0879099010755358) q[1];
cx q[0],q[1];
ry(-2.742155230687397) q[0];
ry(1.1442852495230407) q[1];
cx q[0],q[1];
ry(-1.3902903309669714) q[1];
ry(-1.9084682947308975) q[2];
cx q[1],q[2];
ry(2.370028456383183) q[1];
ry(-1.6777910365422188) q[2];
cx q[1],q[2];
ry(-2.5845100371432848) q[2];
ry(-2.4138182867725373) q[3];
cx q[2],q[3];
ry(2.3841538212495976) q[2];
ry(2.4634270336252317) q[3];
cx q[2],q[3];
ry(0.16499709578990096) q[3];
ry(3.126731159766532) q[4];
cx q[3],q[4];
ry(1.657869029351099) q[3];
ry(-0.6502543461064274) q[4];
cx q[3],q[4];
ry(2.7699323960552302) q[4];
ry(-2.4466632472671916) q[5];
cx q[4],q[5];
ry(1.5952764023203905) q[4];
ry(1.4297254346582435) q[5];
cx q[4],q[5];
ry(1.9726642663164977) q[5];
ry(2.360791931036759) q[6];
cx q[5],q[6];
ry(-2.5795685188459796) q[5];
ry(-0.025728548369933338) q[6];
cx q[5],q[6];
ry(1.674485706487456) q[6];
ry(-2.9544297876603394) q[7];
cx q[6],q[7];
ry(-2.193129455113512) q[6];
ry(2.582658317573207) q[7];
cx q[6],q[7];
ry(2.0428322170712345) q[0];
ry(0.908243793084397) q[1];
cx q[0],q[1];
ry(-1.9552901536435912) q[0];
ry(-2.910601724126782) q[1];
cx q[0],q[1];
ry(1.7600771970224351) q[1];
ry(-0.36596888110695214) q[2];
cx q[1],q[2];
ry(1.846956323120383) q[1];
ry(-2.3525879508922487) q[2];
cx q[1],q[2];
ry(1.899674991179359) q[2];
ry(-2.274441440057209) q[3];
cx q[2],q[3];
ry(-2.9153593332635315) q[2];
ry(-1.1618145684644037) q[3];
cx q[2],q[3];
ry(-2.737758655977612) q[3];
ry(0.8302230915591187) q[4];
cx q[3],q[4];
ry(0.979019584022856) q[3];
ry(0.1820869361155779) q[4];
cx q[3],q[4];
ry(-1.5454499620213469) q[4];
ry(-2.7678086802103135) q[5];
cx q[4],q[5];
ry(2.668955363583606) q[4];
ry(-2.578541939877921) q[5];
cx q[4],q[5];
ry(0.9575881889526637) q[5];
ry(2.307594720187762) q[6];
cx q[5],q[6];
ry(0.8040008838930444) q[5];
ry(1.7000574733237213) q[6];
cx q[5],q[6];
ry(-1.5289144952502705) q[6];
ry(-0.20172876934676648) q[7];
cx q[6],q[7];
ry(-0.7061227517695206) q[6];
ry(2.5310594477622006) q[7];
cx q[6],q[7];
ry(2.3036298848308845) q[0];
ry(-0.5038513048474995) q[1];
cx q[0],q[1];
ry(-2.21691443212712) q[0];
ry(-1.3590048409303055) q[1];
cx q[0],q[1];
ry(1.6646262074164786) q[1];
ry(0.14802945396895628) q[2];
cx q[1],q[2];
ry(2.956923959904205) q[1];
ry(-0.10775279451494058) q[2];
cx q[1],q[2];
ry(0.491978327847471) q[2];
ry(-1.489550055745844) q[3];
cx q[2],q[3];
ry(2.242361819660683) q[2];
ry(-1.8627973220393181) q[3];
cx q[2],q[3];
ry(-2.6479959365441834) q[3];
ry(-0.9187463470408211) q[4];
cx q[3],q[4];
ry(-1.8266400785603418) q[3];
ry(2.1342304098271563) q[4];
cx q[3],q[4];
ry(-2.033050522300562) q[4];
ry(-0.4675206996006375) q[5];
cx q[4],q[5];
ry(-2.0365001985380253) q[4];
ry(-2.3869358768772173) q[5];
cx q[4],q[5];
ry(2.9586967183690547) q[5];
ry(1.9478631508156843) q[6];
cx q[5],q[6];
ry(1.5265650318436972) q[5];
ry(-1.720326323499648) q[6];
cx q[5],q[6];
ry(1.9416846442041962) q[6];
ry(-2.0307844578352734) q[7];
cx q[6],q[7];
ry(2.8651604100672867) q[6];
ry(0.3827873123168523) q[7];
cx q[6],q[7];
ry(-1.5126445837169538) q[0];
ry(1.2811883523974115) q[1];
cx q[0],q[1];
ry(-2.099398353330441) q[0];
ry(-0.8282698536747137) q[1];
cx q[0],q[1];
ry(-2.11205950669411) q[1];
ry(-2.313432449916682) q[2];
cx q[1],q[2];
ry(2.5835978956741745) q[1];
ry(0.5427093080022939) q[2];
cx q[1],q[2];
ry(0.880294411861474) q[2];
ry(-0.14027812497780712) q[3];
cx q[2],q[3];
ry(-1.0641787349378644) q[2];
ry(-2.207654012337768) q[3];
cx q[2],q[3];
ry(0.825372749492427) q[3];
ry(1.105838494564387) q[4];
cx q[3],q[4];
ry(-0.9705180380840792) q[3];
ry(2.616817348488848) q[4];
cx q[3],q[4];
ry(1.7204404043053734) q[4];
ry(1.5335380417967885) q[5];
cx q[4],q[5];
ry(-1.1256791252796736) q[4];
ry(2.0768405521914057) q[5];
cx q[4],q[5];
ry(-2.6076807962420308) q[5];
ry(0.7061025515881958) q[6];
cx q[5],q[6];
ry(-2.5386733431014346) q[5];
ry(0.39924415344828734) q[6];
cx q[5],q[6];
ry(-0.3562035508591191) q[6];
ry(0.4002355647615943) q[7];
cx q[6],q[7];
ry(-2.152150904331305) q[6];
ry(0.9917558716123002) q[7];
cx q[6],q[7];
ry(-3.083113743221404) q[0];
ry(2.276475875459069) q[1];
cx q[0],q[1];
ry(0.29689736853740994) q[0];
ry(2.7253744273590863) q[1];
cx q[0],q[1];
ry(-0.6335173738044614) q[1];
ry(-2.124696653980888) q[2];
cx q[1],q[2];
ry(2.854835174396286) q[1];
ry(2.3622960162124067) q[2];
cx q[1],q[2];
ry(2.440252393089774) q[2];
ry(2.2056335098473285) q[3];
cx q[2],q[3];
ry(0.9699962619865774) q[2];
ry(0.3416773402593112) q[3];
cx q[2],q[3];
ry(0.5557211597243245) q[3];
ry(-3.1413208571426416) q[4];
cx q[3],q[4];
ry(-1.9561496221524728) q[3];
ry(0.6085954999785397) q[4];
cx q[3],q[4];
ry(-2.621996222902523) q[4];
ry(1.356797422869386) q[5];
cx q[4],q[5];
ry(-1.438933157691297) q[4];
ry(-2.0573606653601555) q[5];
cx q[4],q[5];
ry(0.18748265367587372) q[5];
ry(2.9102418911716117) q[6];
cx q[5],q[6];
ry(1.7161842394993023) q[5];
ry(0.6196112320385199) q[6];
cx q[5],q[6];
ry(-1.2825687223509068) q[6];
ry(-0.5391532429663357) q[7];
cx q[6],q[7];
ry(-0.32869198918375275) q[6];
ry(-0.12064895167089766) q[7];
cx q[6],q[7];
ry(0.8562478846325484) q[0];
ry(0.12313512378873323) q[1];
cx q[0],q[1];
ry(-1.0276455901935306) q[0];
ry(-0.2927296386942243) q[1];
cx q[0],q[1];
ry(0.3591496081240752) q[1];
ry(-1.701486866360417) q[2];
cx q[1],q[2];
ry(-1.879214454267938) q[1];
ry(3.11575392251009) q[2];
cx q[1],q[2];
ry(-3.1173056183150494) q[2];
ry(0.8017868525969991) q[3];
cx q[2],q[3];
ry(0.9359648439446218) q[2];
ry(-0.7018239767088812) q[3];
cx q[2],q[3];
ry(1.6590701107735302) q[3];
ry(1.166392475473523) q[4];
cx q[3],q[4];
ry(0.08836322087298054) q[3];
ry(1.0010757879907817) q[4];
cx q[3],q[4];
ry(1.8396344857834934) q[4];
ry(0.9060856941403763) q[5];
cx q[4],q[5];
ry(-1.7199241190066754) q[4];
ry(-1.8916550647200072) q[5];
cx q[4],q[5];
ry(1.0002787178998964) q[5];
ry(-1.806783581598486) q[6];
cx q[5],q[6];
ry(-1.8613567205409192) q[5];
ry(-1.7353265645586988) q[6];
cx q[5],q[6];
ry(2.721872866535591) q[6];
ry(1.7942714380229206) q[7];
cx q[6],q[7];
ry(1.386808180661809) q[6];
ry(-0.9994153305471868) q[7];
cx q[6],q[7];
ry(1.4986669164692004) q[0];
ry(-1.899300157724424) q[1];
cx q[0],q[1];
ry(1.1804171923101068) q[0];
ry(-0.5083224388455797) q[1];
cx q[0],q[1];
ry(3.127693177120146) q[1];
ry(0.9109632084373125) q[2];
cx q[1],q[2];
ry(-1.5033152137542034) q[1];
ry(2.6378573141935475) q[2];
cx q[1],q[2];
ry(1.1887963044406544) q[2];
ry(1.6296973468382066) q[3];
cx q[2],q[3];
ry(0.5070309668264033) q[2];
ry(2.231652947156597) q[3];
cx q[2],q[3];
ry(2.699519066161217) q[3];
ry(1.5853998003815404) q[4];
cx q[3],q[4];
ry(2.8359765848927103) q[3];
ry(-1.2849897285388803) q[4];
cx q[3],q[4];
ry(0.6735811930316952) q[4];
ry(-2.9157848853366306) q[5];
cx q[4],q[5];
ry(0.7761614877629537) q[4];
ry(-2.9349073505697887) q[5];
cx q[4],q[5];
ry(-0.13242486466697911) q[5];
ry(-1.6644830600754312) q[6];
cx q[5],q[6];
ry(1.3848749970849408) q[5];
ry(0.2814588966669582) q[6];
cx q[5],q[6];
ry(2.2109705235237724) q[6];
ry(2.4348349194996906) q[7];
cx q[6],q[7];
ry(2.60316366484103) q[6];
ry(1.4435844602918708) q[7];
cx q[6],q[7];
ry(-0.039420793920667656) q[0];
ry(-0.6493500951789555) q[1];
cx q[0],q[1];
ry(2.9222038096356626) q[0];
ry(-0.7806306678465519) q[1];
cx q[0],q[1];
ry(1.0123962208730832) q[1];
ry(0.13100892412312673) q[2];
cx q[1],q[2];
ry(-6.235369268458094e-05) q[1];
ry(0.8131975955987026) q[2];
cx q[1],q[2];
ry(-0.2541706163692119) q[2];
ry(2.819517783311169) q[3];
cx q[2],q[3];
ry(2.73987293069346) q[2];
ry(-2.979562528123848) q[3];
cx q[2],q[3];
ry(1.66419179636027) q[3];
ry(-2.471330604816958) q[4];
cx q[3],q[4];
ry(0.8413731742136683) q[3];
ry(0.7269205139736572) q[4];
cx q[3],q[4];
ry(1.2833494958658207) q[4];
ry(2.6723638510510503) q[5];
cx q[4],q[5];
ry(-2.40925529346651) q[4];
ry(-1.1657658977722252) q[5];
cx q[4],q[5];
ry(0.5424319613196849) q[5];
ry(0.3938614373970149) q[6];
cx q[5],q[6];
ry(-2.872997928768921) q[5];
ry(-1.5049365150919793) q[6];
cx q[5],q[6];
ry(-1.5306429605867526) q[6];
ry(-2.0922978653076534) q[7];
cx q[6],q[7];
ry(-1.666078263450125) q[6];
ry(0.936908413228604) q[7];
cx q[6],q[7];
ry(-0.5684922839546136) q[0];
ry(1.2321085835520398) q[1];
cx q[0],q[1];
ry(2.0732767297141064) q[0];
ry(0.1280253039050562) q[1];
cx q[0],q[1];
ry(0.13571353316599488) q[1];
ry(-2.9756114574277346) q[2];
cx q[1],q[2];
ry(-2.6695647840003396) q[1];
ry(-1.5953586149470542) q[2];
cx q[1],q[2];
ry(-2.0341053331322003) q[2];
ry(-2.475577580836338) q[3];
cx q[2],q[3];
ry(-3.124081844113273) q[2];
ry(2.776818245088662) q[3];
cx q[2],q[3];
ry(1.639088584254717) q[3];
ry(-1.5801335964792562) q[4];
cx q[3],q[4];
ry(-0.2659303377174087) q[3];
ry(-1.8328024337490687) q[4];
cx q[3],q[4];
ry(-3.067532772567175) q[4];
ry(1.3492606041771498) q[5];
cx q[4],q[5];
ry(-1.6476908428792383) q[4];
ry(0.14279461527361856) q[5];
cx q[4],q[5];
ry(-2.003641884041655) q[5];
ry(1.698667619474149) q[6];
cx q[5],q[6];
ry(0.0694120722516031) q[5];
ry(-1.1694135500132494) q[6];
cx q[5],q[6];
ry(1.8100629577602712) q[6];
ry(-0.19878887970458517) q[7];
cx q[6],q[7];
ry(-1.5702950312949595) q[6];
ry(0.17348544696280213) q[7];
cx q[6],q[7];
ry(-2.870543187043528) q[0];
ry(-2.6712512826722836) q[1];
cx q[0],q[1];
ry(1.7606805389775746) q[0];
ry(-1.0541823539741377) q[1];
cx q[0],q[1];
ry(-2.4979293215741563) q[1];
ry(-1.031138395512394) q[2];
cx q[1],q[2];
ry(2.7674374884919697) q[1];
ry(2.3408812545899775) q[2];
cx q[1],q[2];
ry(-3.0075840695198837) q[2];
ry(-0.5430858571694221) q[3];
cx q[2],q[3];
ry(0.12757395111378525) q[2];
ry(0.0020968105311673924) q[3];
cx q[2],q[3];
ry(-0.48298281047816444) q[3];
ry(-2.87608804474838) q[4];
cx q[3],q[4];
ry(-1.0723688578365487) q[3];
ry(0.06148054363297906) q[4];
cx q[3],q[4];
ry(-2.388294139220663) q[4];
ry(-1.8986484144800588) q[5];
cx q[4],q[5];
ry(-0.9818770648511972) q[4];
ry(2.1171381543795516) q[5];
cx q[4],q[5];
ry(0.9303985312479213) q[5];
ry(1.2002622935422134) q[6];
cx q[5],q[6];
ry(-0.610116127058359) q[5];
ry(1.9102170477115887) q[6];
cx q[5],q[6];
ry(-0.20794294946609604) q[6];
ry(1.6662911813167618) q[7];
cx q[6],q[7];
ry(-0.8078148110204184) q[6];
ry(-2.6324298890549054) q[7];
cx q[6],q[7];
ry(-2.6368088740834037) q[0];
ry(2.173356876799735) q[1];
cx q[0],q[1];
ry(-0.9141263819785131) q[0];
ry(-1.861813131198037) q[1];
cx q[0],q[1];
ry(-1.9041062299079092) q[1];
ry(1.1881548377307234) q[2];
cx q[1],q[2];
ry(0.5078219693802772) q[1];
ry(0.015556006486121987) q[2];
cx q[1],q[2];
ry(1.316936549463585) q[2];
ry(2.978934517295165) q[3];
cx q[2],q[3];
ry(0.7782630552218954) q[2];
ry(0.2419706444312795) q[3];
cx q[2],q[3];
ry(2.555006902819975) q[3];
ry(2.451522042851573) q[4];
cx q[3],q[4];
ry(3.1156661461542554) q[3];
ry(1.0182404390645299) q[4];
cx q[3],q[4];
ry(-2.173941903780011) q[4];
ry(1.4021042642634782) q[5];
cx q[4],q[5];
ry(0.37379984961821977) q[4];
ry(0.5742583779137385) q[5];
cx q[4],q[5];
ry(1.5385263964044809) q[5];
ry(-1.9839834556280662) q[6];
cx q[5],q[6];
ry(-0.6968078888670748) q[5];
ry(1.7963978973552122) q[6];
cx q[5],q[6];
ry(3.072607916833506) q[6];
ry(1.8022827997536348) q[7];
cx q[6],q[7];
ry(0.15478000713594398) q[6];
ry(2.7687883452797317) q[7];
cx q[6],q[7];
ry(1.200889968936047) q[0];
ry(-2.652642506554872) q[1];
cx q[0],q[1];
ry(-2.383583471854546) q[0];
ry(-0.5905556823496401) q[1];
cx q[0],q[1];
ry(2.149180597039015) q[1];
ry(2.4678611457669533) q[2];
cx q[1],q[2];
ry(-0.7610979694995965) q[1];
ry(-3.026654017789954) q[2];
cx q[1],q[2];
ry(-0.08048720687103293) q[2];
ry(2.504317724745926) q[3];
cx q[2],q[3];
ry(2.3617728164051286) q[2];
ry(-2.8440781303120106) q[3];
cx q[2],q[3];
ry(-2.7743685187086666) q[3];
ry(-2.936581447325911) q[4];
cx q[3],q[4];
ry(-2.5937313817125176) q[3];
ry(-3.134286909209525) q[4];
cx q[3],q[4];
ry(1.6253540619727367) q[4];
ry(1.3211472186464193) q[5];
cx q[4],q[5];
ry(-2.954353411976405) q[4];
ry(0.11963618534931825) q[5];
cx q[4],q[5];
ry(0.8558823923642185) q[5];
ry(-0.1967294651374852) q[6];
cx q[5],q[6];
ry(2.614548254810334) q[5];
ry(-0.12325915392415077) q[6];
cx q[5],q[6];
ry(-1.3462119748446166) q[6];
ry(-1.3169568116278483) q[7];
cx q[6],q[7];
ry(2.394333348797101) q[6];
ry(1.7472119569708504) q[7];
cx q[6],q[7];
ry(-0.2272201871594298) q[0];
ry(-2.191563380996505) q[1];
cx q[0],q[1];
ry(1.0075541888240949) q[0];
ry(2.6816670437094423) q[1];
cx q[0],q[1];
ry(-0.5651122215629966) q[1];
ry(-0.9139338619151509) q[2];
cx q[1],q[2];
ry(-0.14582915762034876) q[1];
ry(2.0705979971541417) q[2];
cx q[1],q[2];
ry(-1.3467554024515067) q[2];
ry(-1.2141594500180546) q[3];
cx q[2],q[3];
ry(2.160783660860728) q[2];
ry(-2.7830882506027543) q[3];
cx q[2],q[3];
ry(0.27757075707353157) q[3];
ry(3.077465840567495) q[4];
cx q[3],q[4];
ry(1.1684090576648927) q[3];
ry(1.786692633226082) q[4];
cx q[3],q[4];
ry(-1.1694041994473547) q[4];
ry(0.7691333217778533) q[5];
cx q[4],q[5];
ry(-1.0480904565456486) q[4];
ry(1.8603841343892957) q[5];
cx q[4],q[5];
ry(0.09670530383887943) q[5];
ry(2.9887545221848364) q[6];
cx q[5],q[6];
ry(-0.5323571167711939) q[5];
ry(0.10031878638392211) q[6];
cx q[5],q[6];
ry(0.5716196209590095) q[6];
ry(-2.2810207037258987) q[7];
cx q[6],q[7];
ry(0.13720936609040546) q[6];
ry(-1.5813647148697878) q[7];
cx q[6],q[7];
ry(0.05966202382481906) q[0];
ry(-0.38776201018417117) q[1];
cx q[0],q[1];
ry(-0.7757113093104158) q[0];
ry(-0.18077240768439484) q[1];
cx q[0],q[1];
ry(2.5673672886158925) q[1];
ry(-1.53409062758503) q[2];
cx q[1],q[2];
ry(-3.0881467711833377) q[1];
ry(1.5820977972412358) q[2];
cx q[1],q[2];
ry(2.328164911004947) q[2];
ry(-1.7295583186522903) q[3];
cx q[2],q[3];
ry(1.2732953395362412) q[2];
ry(-2.4519386349728447) q[3];
cx q[2],q[3];
ry(1.971653277764438) q[3];
ry(-0.7960446414959645) q[4];
cx q[3],q[4];
ry(-2.801714446822758) q[3];
ry(-0.8025377910614286) q[4];
cx q[3],q[4];
ry(-1.4243883782228597) q[4];
ry(1.6246339622382706) q[5];
cx q[4],q[5];
ry(3.0667635824594868) q[4];
ry(1.0157965902203872) q[5];
cx q[4],q[5];
ry(-1.173984617684905) q[5];
ry(-1.1050022691483263) q[6];
cx q[5],q[6];
ry(1.6170901216490572) q[5];
ry(-1.26910622367616) q[6];
cx q[5],q[6];
ry(3.0286645309702247) q[6];
ry(-0.30036181183514904) q[7];
cx q[6],q[7];
ry(-0.7595398462191988) q[6];
ry(1.9550955249755262) q[7];
cx q[6],q[7];
ry(2.6124630742294537) q[0];
ry(1.295186086182028) q[1];
cx q[0],q[1];
ry(-2.165698602773419) q[0];
ry(-1.4521108244311982) q[1];
cx q[0],q[1];
ry(2.665434606153714) q[1];
ry(-0.7611294811467205) q[2];
cx q[1],q[2];
ry(1.1196494551432432) q[1];
ry(2.208283957382075) q[2];
cx q[1],q[2];
ry(3.0056361860036636) q[2];
ry(2.0832180920548593) q[3];
cx q[2],q[3];
ry(-2.942857844949941) q[2];
ry(-1.6075505752355663) q[3];
cx q[2],q[3];
ry(2.1616033342453695) q[3];
ry(0.5857801621511447) q[4];
cx q[3],q[4];
ry(-2.5261550628652496) q[3];
ry(-2.0439968407733535) q[4];
cx q[3],q[4];
ry(-1.8118524636279967) q[4];
ry(-0.1765630451481841) q[5];
cx q[4],q[5];
ry(-2.426988260522484) q[4];
ry(0.5549232997443179) q[5];
cx q[4],q[5];
ry(3.0988230757812123) q[5];
ry(0.9184820531401399) q[6];
cx q[5],q[6];
ry(-2.7541263186427862) q[5];
ry(-0.89222210303205) q[6];
cx q[5],q[6];
ry(-0.3013786243932329) q[6];
ry(-0.14322243523777486) q[7];
cx q[6],q[7];
ry(0.6497961764047834) q[6];
ry(-1.1076361281642093) q[7];
cx q[6],q[7];
ry(-2.729247896309532) q[0];
ry(2.5724396229221034) q[1];
cx q[0],q[1];
ry(-0.8747207581395883) q[0];
ry(-1.173723036154871) q[1];
cx q[0],q[1];
ry(-2.9334554801099566) q[1];
ry(-1.329335417529667) q[2];
cx q[1],q[2];
ry(2.709351627039954) q[1];
ry(-2.88125638731907) q[2];
cx q[1],q[2];
ry(-2.896847768673959) q[2];
ry(-1.4146573473865676) q[3];
cx q[2],q[3];
ry(-2.195370833843472) q[2];
ry(-0.9011918250135763) q[3];
cx q[2],q[3];
ry(-0.5057364881060822) q[3];
ry(-1.1707406580560553) q[4];
cx q[3],q[4];
ry(-0.6367430136747485) q[3];
ry(1.6003125831608447) q[4];
cx q[3],q[4];
ry(1.7640811809612205) q[4];
ry(1.127315683777593) q[5];
cx q[4],q[5];
ry(-2.620215795601644) q[4];
ry(-0.2969055201801183) q[5];
cx q[4],q[5];
ry(0.6370050396956665) q[5];
ry(-1.8434300540997857) q[6];
cx q[5],q[6];
ry(3.077670320143687) q[5];
ry(0.6226268591115071) q[6];
cx q[5],q[6];
ry(2.2072611033842904) q[6];
ry(2.455365645107233) q[7];
cx q[6],q[7];
ry(-0.30302940538448747) q[6];
ry(-1.9088057255329103) q[7];
cx q[6],q[7];
ry(1.8509224856090114) q[0];
ry(1.1490982296863228) q[1];
cx q[0],q[1];
ry(0.2217356712565577) q[0];
ry(-2.8187573547290516) q[1];
cx q[0],q[1];
ry(0.8185323077999936) q[1];
ry(-2.4580660448969383) q[2];
cx q[1],q[2];
ry(-2.5284320761238828) q[1];
ry(-2.71250339333795) q[2];
cx q[1],q[2];
ry(-1.2681311617158242) q[2];
ry(-1.7212105920940362) q[3];
cx q[2],q[3];
ry(1.1085298337641423) q[2];
ry(-0.47357090503046595) q[3];
cx q[2],q[3];
ry(1.480187288460845) q[3];
ry(1.0172123365308803) q[4];
cx q[3],q[4];
ry(1.4420914423793043) q[3];
ry(-1.2872546278895491) q[4];
cx q[3],q[4];
ry(0.7960419954642646) q[4];
ry(-1.1295163449631023) q[5];
cx q[4],q[5];
ry(2.4923740645387675) q[4];
ry(1.3926945925993928) q[5];
cx q[4],q[5];
ry(-0.25313414218363456) q[5];
ry(-2.1515692684757743) q[6];
cx q[5],q[6];
ry(0.3091516700601514) q[5];
ry(-2.8029643122070604) q[6];
cx q[5],q[6];
ry(-0.5175682833301039) q[6];
ry(-2.9492475147549926) q[7];
cx q[6],q[7];
ry(-1.1390532148856218) q[6];
ry(3.054358290441343) q[7];
cx q[6],q[7];
ry(-0.9838356499903339) q[0];
ry(2.009844228923899) q[1];
cx q[0],q[1];
ry(0.3920268077545632) q[0];
ry(0.10065393909148226) q[1];
cx q[0],q[1];
ry(1.745186145221778) q[1];
ry(0.290128861256961) q[2];
cx q[1],q[2];
ry(2.120893643740817) q[1];
ry(0.5125912605449223) q[2];
cx q[1],q[2];
ry(-2.995791701963623) q[2];
ry(1.0493062505160333) q[3];
cx q[2],q[3];
ry(0.924642391335413) q[2];
ry(-0.44186453566145806) q[3];
cx q[2],q[3];
ry(-2.7268727165325677) q[3];
ry(-0.060673875229394145) q[4];
cx q[3],q[4];
ry(-1.7670635416414262) q[3];
ry(-1.015324546883007) q[4];
cx q[3],q[4];
ry(-2.8375681035411424) q[4];
ry(0.9921142982375057) q[5];
cx q[4],q[5];
ry(-2.0769330655859095) q[4];
ry(-0.47394950005319014) q[5];
cx q[4],q[5];
ry(-0.8441446502143193) q[5];
ry(2.8467611383481457) q[6];
cx q[5],q[6];
ry(-0.7174183094378783) q[5];
ry(-2.282711940099533) q[6];
cx q[5],q[6];
ry(-0.8704524464595229) q[6];
ry(-2.1663234599054992) q[7];
cx q[6],q[7];
ry(1.5721612596210113) q[6];
ry(1.921125638945411) q[7];
cx q[6],q[7];
ry(0.3773883758549761) q[0];
ry(-0.5427614599575963) q[1];
ry(1.6619219688711435) q[2];
ry(0.9856257031082672) q[3];
ry(3.048532679150951) q[4];
ry(0.019850954984837063) q[5];
ry(2.5658337200779036) q[6];
ry(-1.8000163839677101) q[7];