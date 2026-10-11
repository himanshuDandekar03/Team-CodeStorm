/**
 * OrbitGuard AI — Guided 72-Hour Collision & Avoidance Demonstration for Rajesh
 * Flow-by-flow interactive test case:
 * Step 1: 72h Conjunction Risk Warning
 * Step 2: Isolated 3D Two-Satellite Trajectory
 * Step 3: Satellite Selection & Maneuver Strategy
 * Step 4: 3D Real-Time Trajectory Deflection
 * Step 5: Fuel Optimization Telemetry & Thruster Parameters
 * Step 6: Executive Mission Summary
 */

class RajeshCollisionDemo {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 6;
        this.selectedSatellite = 'sat1'; // 'sat1' (ASTRA-SAT 1) or 'sat2' (COSMOS-2412)
        this.selectedManeuver = 'radial_outward';
        this.selectedLeadTime = 24.0;
        this.maneuverApplied = false;
        this.scenarioData = null;

        // Three.js instances
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.animationId = null;
        this.sat1Mesh = null;
        this.sat2Mesh = null;
        this.orbit1Line = null;
        this.orbit2Line = null;
        this.deflectedOrbitLine = null;
        this.collisionMarker = null;
        this.safeMarker = null;
        this.simTime = 0;
        this.animSpeed = 0.008;

        // Mathematical model parameters (LEO 550 km)
        this.R_EARTH = 6371.0;
        this.ALT_KM = 550.0;
        this.SCALE = 1.0 / 1000.0; // 1 unit = 1000 km
        this.rOrbit = (this.R_EARTH + this.ALT_KM) * this.SCALE;
        this.rEarthScaled = this.R_EARTH * this.SCALE;

        // Collision Unit Vector (Latitude 42.36° N, 45.0° azimuth in X-Z plane, facing camera directly)
        // [cos(42.36°)*cos(45°), sin(42.36°), cos(42.36°)*sin(45°)]
        this.uColl = new THREE.Vector3(0.52249952, 0.67378668, 0.52249952);
        this.collisionPoint = this.uColl.clone().multiplyScalar(this.rOrbit);

        // Orthonormal orbital bases: both Orbit 1 (Inc: 53.0°) and Orbit 2 (Inc: 86.4°) pass through this.uColl at theta = 0
        this.e1_1 = this.uColl.clone();
        this.e1_2 = new THREE.Vector3(-0.85234901, 0.42875422, 0.29945112);

        this.e2_1 = this.uColl.clone();
        this.e2_2 = new THREE.Vector3(0.41462920, -0.73625325, 0.53480255);
    }

    getOrbitPoint(orbitIndex, theta, r = this.rOrbit) {
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        const e1 = (orbitIndex === 1) ? this.e1_1 : this.e2_1;
        const e2 = (orbitIndex === 1) ? this.e1_2 : this.e2_2;
        return new THREE.Vector3(
            r * (cosT * e1.x + sinT * e2.x),
            r * (cosT * e1.y + sinT * e2.y),
            r * (cosT * e1.z + sinT * e2.z)
        );
    }

    getOrbitVelocity(orbitIndex, theta) {
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        const e1 = (orbitIndex === 1) ? this.e1_1 : this.e2_1;
        const e2 = (orbitIndex === 1) ? this.e1_2 : this.e2_2;
        return new THREE.Vector3(
            -sinT * e1.x + cosT * e2.x,
            -sinT * e1.y + cosT * e2.y,
            -sinT * e1.z + cosT * e2.z
        ).normalize();
    }

    async init() {
        try {
            const res = await fetch('/api/demo/rajesh-collision');
            if (res.ok) {
                const data = await res.json();
                if (data && data.success) {
                    this.scenarioData = data.scenario;
                }
            }
        } catch (err) {
            console.warn('Using built-in scenario fallback for Rajesh demo:', err);
        }
        if (!this.scenarioData) {
            this.scenarioData = this.getDefaultScenario();
        }
    }

    getDefaultScenario() {
        return {
            satellite_1: {
                id: 'ASTRA-SAT 1',
                norad_id: 48210,
                type: 'Active Earth Observation Spacecraft',
                altitude_km: 550.0,
                inclination_deg: 53.0,
                period_min: 95.6,
                velocity_km_s: 7.58,
                mass_kg: 1200.0,
                fuel_capacity_kg: 85.0,
                isp_s: 300.0,
                maneuverable: true,
                color: '#00f0ff'
            },
            satellite_2: {
                id: 'COSMOS-2412 DEBRIS',
                norad_id: 38914,
                type: 'Non-Cooperative Defunct Fragment',
                altitude_km: 553.0,
                inclination_deg: 86.4,
                period_min: 95.7,
                velocity_km_s: 7.62,
                mass_kg: 450.0,
                fuel_capacity_kg: 0.0,
                isp_s: 0.0,
                maneuverable: false,
                color: '#ff3366'
            },
            collision: {
                time_to_impact_hours: 72.0,
                tca_utc: 'T + 72h 00m 00s',
                miss_distance_km: 0.00,
                relative_velocity_km_s: 10.74,
                collision_probability: 0.987,
                risk_level: 'CRITICAL RED'
            },
            maneuvers: {
                radial_outward: {
                    name: 'Optimal Radial Outward (+R)',
                    direction: '+R (Radial Outward)',
                    delta_v_m_s: 0.38,
                    burn_lead_time_hours: 24.0,
                    miss_distance_km: 52.4,
                    fuel_consumed_kg: 0.155,
                    fuel_reserve_percent: 99.82,
                    risk_level: 'GREEN'
                },
                prograde: {
                    name: 'In-Track Prograde (+V)',
                    direction: '+T (In-Track Prograde)',
                    delta_v_m_s: 0.54,
                    burn_lead_time_hours: 24.0,
                    miss_distance_km: 64.1,
                    fuel_consumed_kg: 0.220,
                    fuel_reserve_percent: 99.74,
                    risk_level: 'GREEN'
                },
                retrograde: {
                    name: 'In-Track Retrograde (-V)',
                    direction: '-T (In-Track Retrograde)',
                    delta_v_m_s: 0.48,
                    burn_lead_time_hours: 24.0,
                    miss_distance_km: 58.7,
                    fuel_consumed_kg: 0.196,
                    fuel_reserve_percent: 99.77,
                    risk_level: 'GREEN'
                }
            }
        };
    }

    launch() {
        const modal = document.getElementById('rajesh-demo-modal');
        if (!modal) {
            console.error('Rajesh demo modal element not found');
            return;
        }
        modal.style.display = 'flex';
        this.currentStep = 1;
        this.maneuverApplied = false;
        this.selectedSatellite = 'sat1';
        this.selectedManeuver = 'radial_outward';
        this.renderStep();
    }

    close() {
        const modal = document.getElementById('rajesh-demo-modal');
        if (modal) modal.style.display = 'none';
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    nextStep() {
        if (this.currentStep < this.totalSteps) {
            if (this.currentStep === 3) {
                // If on Step 3 and advancing, apply maneuver automatically
                this.applyManeuver();
            }
            this.currentStep++;
            this.renderStep();
        }
    }

    prevStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.renderStep();
        }
    }

    goToStep(stepNum) {
        if (stepNum >= 1 && stepNum <= this.totalSteps) {
            if (stepNum >= 4 && !this.maneuverApplied) {
                this.applyManeuver();
            }
            this.currentStep = stepNum;
            this.renderStep();
        }
    }

    renderStep() {
        // Update step pills
        for (let i = 1; i <= this.totalSteps; i++) {
            const pill = document.getElementById(`rajesh-step-pill-${i}`);
            if (pill) {
                pill.classList.remove('active', 'completed');
                if (i === this.currentStep) {
                    pill.classList.add('active');
                } else if (i < this.currentStep) {
                    pill.classList.add('completed');
                }
            }
        }

        // Hide all step sections
        for (let i = 1; i <= this.totalSteps; i++) {
            const sec = document.getElementById(`rajesh-step-${i}`);
            if (sec) sec.style.display = 'none';
        }

        // Show current step section
        const activeSec = document.getElementById(`rajesh-step-${this.currentStep}`);
        if (activeSec) activeSec.style.display = 'block';

        // Update nav buttons
        const prevBtn = document.getElementById('rajesh-prev-btn');
        const nextBtn = document.getElementById('rajesh-next-btn');
        if (prevBtn) {
            prevBtn.disabled = (this.currentStep === 1);
            prevBtn.style.opacity = (this.currentStep === 1) ? '0.4' : '1';
        }
        if (nextBtn) {
            const nextLabels = {
                1: 'See 3D Trajectory on 3D Model ➡',
                2: 'Next: Plan Avoidance & Select Satellite ➡',
                3: 'Apply Maneuver & Change Path ➡',
                4: 'Next: Fuel Optimization Parameters ➡',
                5: 'Next: Executive Mission Summary ➡',
                6: 'Done: Return to Mission Control'
            };
            nextBtn.innerHTML = `<span>${nextLabels[this.currentStep] || 'Next ➡'}</span>`;
            if (this.currentStep === 6) {
                nextBtn.onclick = () => this.close();
            } else {
                nextBtn.onclick = () => this.nextStep();
            }
        }

        // Step-specific initializations
        if (this.currentStep === 2 || this.currentStep === 4) {
            setTimeout(() => {
                this.initThreeScene();
                this.updateThreeSceneForStep(this.currentStep);
            }, 50);
        } else {
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
                this.animationId = null;
            }
        }
    }

    selectSatellite(satKey) {
        this.selectedSatellite = satKey;
        const sat1Card = document.getElementById('rajesh-sat-card-1');
        const sat2Card = document.getElementById('rajesh-sat-card-2');
        const warningBox = document.getElementById('rajesh-sat-warning');

        if (satKey === 'sat1') {
            if (sat1Card) sat1Card.classList.add('selected');
            if (sat2Card) sat2Card.classList.remove('selected');
            if (warningBox) warningBox.style.display = 'none';
        } else {
            if (sat1Card) sat1Card.classList.remove('selected');
            if (sat2Card) sat2Card.classList.add('selected');
            if (warningBox) {
                warningBox.style.display = 'block';
                warningBox.innerHTML = `
                    <i class="fas fa-exclamation-triangle" style="color: #ffb300; margin-right: 8px;"></i>
                    <strong>Operational Constraint:</strong> COSMOS-2412 is a non-cooperative defunct fragment with zero propulsion. 
                    In real orbital safety protocols, only active spacecraft (ASTRA-SAT 1) can execute maneuvers. 
                    Switching back to ASTRA-SAT 1 is strongly recommended.
                `;
            }
        }
    }

    selectManeuver(stratKey) {
        this.selectedManeuver = stratKey;
        ['radial_outward', 'prograde', 'retrograde'].forEach(key => {
            const btn = document.getElementById(`rajesh-strat-${key}`);
            if (btn) {
                if (key === stratKey) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            }
        });
        this.updateManeuverPreview();
    }

    updateManeuverPreview() {
        const strat = this.scenarioData.maneuvers[this.selectedManeuver] || this.scenarioData.maneuvers['radial_outward'];
        const previewEl = document.getElementById('rajesh-maneuver-preview');
        if (previewEl) {
            previewEl.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 10px;">
                    <div class="hud-stat">
                        <span class="hud-label">Impulsive ΔV</span>
                        <span class="hud-val" style="color: #00f0ff;">${strat.delta_v_m_s.toFixed(2)} m/s</span>
                    </div>
                    <div class="hud-stat">
                        <span class="hud-label">Propellant Mass</span>
                        <span class="hud-val" style="color: #00e676;">${strat.fuel_consumed_kg.toFixed(3)} kg</span>
                    </div>
                    <div class="hud-stat">
                        <span class="hud-label">Expected Miss Distance</span>
                        <span class="hud-val" style="color: #00e676;">${strat.miss_distance_km.toFixed(1)} km</span>
                    </div>
                    <div class="hud-stat">
                        <span class="hud-label">Safety Status</span>
                        <span class="hud-val" style="color: #00e676;">FEASIBLE (CLEAR)</span>
                    </div>
                </div>
            `;
        }
    }

    applyManeuver() {
        this.maneuverApplied = true;
        const strat = this.scenarioData.maneuvers[this.selectedManeuver] || this.scenarioData.maneuvers['radial_outward'];

        // Update Step 4 telemetry
        const step4Miss = document.getElementById('step4-new-miss');
        if (step4Miss) step4Miss.textContent = `${strat.miss_distance_km.toFixed(1)} km`;

        const step4Dv = document.getElementById('step4-applied-dv');
        if (step4Dv) step4Dv.textContent = `${strat.delta_v_m_s.toFixed(2)} m/s (${strat.direction})`;

        // Update Step 5 telemetry
        const step5Dv = document.getElementById('step5-dv');
        if (step5Dv) step5Dv.textContent = `${strat.delta_v_m_s.toFixed(2)} m/s`;

        const step5Fuel = document.getElementById('step5-fuel');
        if (step5Fuel) step5Fuel.textContent = `${strat.fuel_consumed_kg.toFixed(3)} kg (${(strat.fuel_consumed_kg * 1000).toFixed(0)} g)`;

        const step5Reserve = document.getElementById('step5-reserve');
        if (step5Reserve) step5Reserve.textContent = `${(85.0 - strat.fuel_consumed_kg).toFixed(3)} kg / 85.0 kg (${strat.fuel_reserve_percent}%)`;

        // If 3D is active, update orbit geometry
        if (this.scene) {
            this.updateThreeSceneForStep(4);
        }
    }

    /* ══════════════════════════════════════════════════════════════════════
       THREE.JS DEDICATED 3D ENGINE FOR RAJESH DEMO
       Renders ONLY the two colliding satellites and their orbital paths
       ══════════════════════════════════════════════════════════════════════ */
    initThreeScene() {
        const container = document.getElementById('rajesh-3d-container');
        if (!container) return;

        // If renderer already attached to this container, just resize
        if (this.renderer && this.renderer.domElement.parentElement === container) {
            this.resizeThree();
            return;
        }

        // Clean existing
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }

        const width = container.clientWidth || 800;
        const height = container.clientHeight || 460;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x040814);

        // Starfield background
        const starGeo = new THREE.BufferGeometry();
        const starCount = 1200;
        const starPositions = new Float32Array(starCount * 3);
        for (let i = 0; i < starCount * 3; i += 3) {
            starPositions[i] = (Math.random() - 0.5) * 60;
            starPositions[i + 1] = (Math.random() - 0.5) * 60;
            starPositions[i + 2] = (Math.random() - 0.5) * 60;
        }
        starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
        const starMat = new THREE.PointsMaterial({ color: 0x88bbff, size: 0.12, transparent: true, opacity: 0.8 });
        const starField = new THREE.Points(starGeo, starMat);
        this.scene.add(starField);

        // Camera
        this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
        this.camera.position.set(11, 7, 12);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        // Controls
        if (THREE.OrbitControls) {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.08;
            this.controls.minDistance = 8;
            this.controls.maxDistance = 30;
            this.controls.target.set(0, 0, 0);
        }

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x223355, 1.2);
        this.scene.add(ambientLight);

        const sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
        sunLight.position.set(20, 15, 20);
        this.scene.add(sunLight);

        const rimLight = new THREE.DirectionalLight(0x00f0ff, 0.6);
        rimLight.position.set(-15, -10, -15);
        this.scene.add(rimLight);

        // Realistic Earth Globe
        this.createEarth();

        // Build the Orbits
        this.createOrbits();

        // Build Satellite Models
        this.createSatellites();

        // Build Collision Marker
        this.createCollisionMarker();

        // Start animation loop
        this.animate();

        // Window resize handler
        window.addEventListener('resize', () => this.resizeThree());
    }

    createEarth() {
        const earthRadius = this.rEarthScaled;
        const earthGeo = new THREE.SphereGeometry(earthRadius, 48, 48);
        const earthMat = new THREE.MeshPhongMaterial({
            color: 0x1a457a,
            emissive: 0x051329,
            specular: 0x3377bb,
            shininess: 25,
            wireframe: false
        });
        const earth = new THREE.Mesh(earthGeo, earthMat);
        this.scene.add(earth);

        // Atmosphere glow
        const atmoGeo = new THREE.SphereGeometry(earthRadius * 1.02, 48, 48);
        const atmoMat = new THREE.MeshBasicMaterial({
            color: 0x4da3ff,
            transparent: true,
            opacity: 0.18,
            side: THREE.BackSide
        });
        const atmo = new THREE.Mesh(atmoGeo, atmoMat);
        this.scene.add(atmo);

        // Latitude & Longitude grid lines
        const gridGeo = new THREE.SphereGeometry(earthRadius * 1.004, 24, 24);
        const gridMat = new THREE.MeshBasicMaterial({
            color: 0x00f0ff,
            wireframe: true,
            transparent: true,
            opacity: 0.12
        });
        const grid = new THREE.Mesh(gridGeo, gridMat);
        this.scene.add(grid);

        // Equator ring
        const eqGeo = new THREE.RingGeometry(earthRadius * 1.005, earthRadius * 1.015, 64);
        const eqMat = new THREE.MeshBasicMaterial({ color: 0x00ffff, side: THREE.DoubleSide, transparent: true, opacity: 0.3 });
        const eq = new THREE.Mesh(eqGeo, eqMat);
        eq.rotation.x = Math.PI / 2;
        this.scene.add(eq);
    }

    createOrbits() {
        const r = this.rOrbit;
        const segments = 240;

        // Orbit 1: ASTRA-SAT 1 (Cyan line, Inclination 53.0°)
        const orbit1Points = [];
        for (let i = 0; i <= segments; i++) {
            const theta = (i / segments) * Math.PI * 2;
            orbit1Points.push(this.getOrbitPoint(1, theta, r));
        }
        const orbit1Geo = new THREE.BufferGeometry().setFromPoints(orbit1Points);
        const orbit1Mat = new THREE.LineBasicMaterial({
            color: 0x00f0ff,
            linewidth: 2,
            transparent: true,
            opacity: 0.85
        });
        this.orbit1Line = new THREE.Line(orbit1Geo, orbit1Mat);
        this.scene.add(this.orbit1Line);

        // Orbit 2: COSMOS-2412 DEBRIS (Red line, Inclination 86.4°)
        // Passes PRECISELY through Orbit 1 at theta = 0 (this.collisionPoint)
        const orbit2Points = [];
        for (let i = 0; i <= segments; i++) {
            const theta = (i / segments) * Math.PI * 2;
            orbit2Points.push(this.getOrbitPoint(2, theta, r));
        }
        const orbit2Geo = new THREE.BufferGeometry().setFromPoints(orbit2Points);
        const orbit2Mat = new THREE.LineBasicMaterial({
            color: 0xff3366,
            linewidth: 2,
            transparent: true,
            opacity: 0.85
        });
        this.orbit2Line = new THREE.Line(orbit2Geo, orbit2Mat);
        this.scene.add(this.orbit2Line);

        // Deflected Orbit (for Step 4) — Glowing Green deflected trajectory
        const deflectedPoints = [];
        const deltaR = 0.524; // Scaled 52.4 km miss distance clearance
        for (let i = 0; i <= segments; i++) {
            const theta = (i / segments) * Math.PI * 2;
            // Radial displacement centered at the conjunction encounter (theta = 0)
            const phaseDist = Math.min(theta, Math.PI * 2 - theta);
            const radialKick = deltaR * Math.exp(-(phaseDist * phaseDist) / 1.2);
            const rEff = r + radialKick;
            deflectedPoints.push(this.getOrbitPoint(1, theta, rEff));
        }
        const deflectedGeo = new THREE.BufferGeometry().setFromPoints(deflectedPoints);
        const deflectedMat = new THREE.LineBasicMaterial({
            color: 0x00e676,
            linewidth: 3,
            transparent: true,
            opacity: 0.95
        });
        this.deflectedOrbitLine = new THREE.Line(deflectedGeo, deflectedMat);
        this.deflectedOrbitLine.visible = false;
        this.scene.add(this.deflectedOrbitLine);
    }

    createSatellites() {
        // Satellite 1: ASTRA-SAT 1 (Cyan Spacecraft with solar wings)
        const sat1Group = new THREE.Group();
        const bus1 = new THREE.Mesh(
            new THREE.BoxGeometry(0.25, 0.25, 0.35),
            new THREE.MeshStandardMaterial({ color: 0x00f0ff, metalness: 0.8, roughness: 0.2 })
        );
        sat1Group.add(bus1);

        // Solar panels
        const panelGeo = new THREE.BoxGeometry(0.7, 0.03, 0.25);
        const panelMat = new THREE.MeshStandardMaterial({ color: 0x1144aa, roughness: 0.3 });
        const leftPanel = new THREE.Mesh(panelGeo, panelMat);
        leftPanel.position.x = -0.55;
        sat1Group.add(leftPanel);

        const rightPanel = new THREE.Mesh(panelGeo, panelMat);
        rightPanel.position.x = 0.55;
        sat1Group.add(rightPanel);

        // Beacon light
        const sat1Light = new THREE.PointLight(0x00f0ff, 2, 3);
        sat1Group.add(sat1Light);

        this.sat1Mesh = sat1Group;
        this.scene.add(this.sat1Mesh);

        // Satellite 2: COSMOS-2412 DEBRIS (Red fragmented geometry)
        const sat2Group = new THREE.Group();
        const debrisBus = new THREE.Mesh(
            new THREE.DodecahedronGeometry(0.24, 0),
            new THREE.MeshStandardMaterial({ color: 0xff3366, roughness: 0.8, metalness: 0.4 })
        );
        sat2Group.add(debrisBus);

        const sat2Light = new THREE.PointLight(0xff3366, 2, 3);
        sat2Group.add(sat2Light);

        this.sat2Mesh = sat2Group;
        this.scene.add(this.sat2Mesh);
    }

    createCollisionMarker() {
        const markerGroup = new THREE.Group();

        // Pulsing outer ring
        const ringGeo = new THREE.RingGeometry(0.32, 0.48, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xff0044, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        markerGroup.add(ring);

        // Inner reticle ring
        const innerRingGeo = new THREE.RingGeometry(0.12, 0.18, 24);
        const innerRingMat = new THREE.MeshBasicMaterial({ color: 0xff3366, side: THREE.DoubleSide, transparent: true, opacity: 0.9 });
        const innerRing = new THREE.Mesh(innerRingGeo, innerRingMat);
        markerGroup.add(innerRing);

        // Center collision dot
        const dotGeo = new THREE.SphereGeometry(0.09, 16, 16);
        const dotMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const dot = new THREE.Mesh(dotGeo, dotMat);
        markerGroup.add(dot);

        // Crosshairs for high-tech HUD targeting appearance
        const crosshairMat = new THREE.LineBasicMaterial({ color: 0xff3366, transparent: true, opacity: 0.8 });
        const chGeoH = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.55, 0, 0), new THREE.Vector3(0.55, 0, 0)]);
        const chH = new THREE.Line(chGeoH, crosshairMat);
        markerGroup.add(chH);

        const chGeoV = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -0.55, 0), new THREE.Vector3(0, 0.55, 0)]);
        const chV = new THREE.Line(chGeoV, crosshairMat);
        markerGroup.add(chV);

        // Position PRECISELY at the physical orbital intersection point
        markerGroup.position.copy(this.collisionPoint);
        markerGroup.lookAt(new THREE.Vector3(0, 0, 0));

        this.collisionMarker = markerGroup;
        this.scene.add(this.collisionMarker);

        // Safe clearance marker (for Step 4)
        const safeGroup = new THREE.Group();
        const safeRingGeo = new THREE.RingGeometry(0.32, 0.52, 32);
        const safeRingMat = new THREE.MeshBasicMaterial({ color: 0x00e676, side: THREE.DoubleSide, transparent: true, opacity: 0.9 });
        const safeRing = new THREE.Mesh(safeRingGeo, safeRingMat);
        safeGroup.add(safeRing);

        const safeDot = new THREE.Mesh(new THREE.SphereGeometry(0.10, 16, 16), new THREE.MeshBasicMaterial({ color: 0x00e676 }));
        safeGroup.add(safeDot);

        const deltaR = 0.524;
        safeGroup.position.copy(this.uColl).multiplyScalar(this.rOrbit + deltaR);
        safeGroup.lookAt(new THREE.Vector3(0, 0, 0));
        safeGroup.visible = false;

        this.safeMarker = safeGroup;
        this.scene.add(this.safeMarker);
    }

    updateThreeSceneForStep(step) {
        if (!this.scene) return;

        if (step === 2) {
            // STEP 2: Collision Trajectory (Before Avoidance)
            if (this.orbit1Line) {
                this.orbit1Line.material.color.setHex(0x00f0ff);
                this.orbit1Line.material.opacity = 0.85;
                this.orbit1Line.visible = true;
            }
            if (this.orbit2Line) {
                this.orbit2Line.material.color.setHex(0xff3366);
                this.orbit2Line.visible = true;
            }
            if (this.deflectedOrbitLine) {
                this.deflectedOrbitLine.visible = false;
            }
            if (this.collisionMarker) {
                this.collisionMarker.visible = true;
            }
            if (this.safeMarker) {
                this.safeMarker.visible = false;
            }
        } else if (step === 4) {
            // STEP 4: Deflected Trajectory (After Avoidance)
            if (this.orbit1Line) {
                // Dim the old collision path to dashed red/ghost
                this.orbit1Line.material.color.setHex(0x553344);
                this.orbit1Line.material.opacity = 0.35;
            }
            if (this.deflectedOrbitLine) {
                this.deflectedOrbitLine.visible = true; // Show glowing green deflected path
            }
            if (this.collisionMarker) {
                this.collisionMarker.visible = false; // Collision removed!
            }
            if (this.safeMarker) {
                this.safeMarker.visible = true; // Show green clearance marker!
            }
        }
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        this.simTime += this.animSpeed;

        // Animate pulse on collision or safe marker
        if (this.collisionMarker && this.collisionMarker.visible) {
            const scale = 1.0 + 0.25 * Math.sin(this.simTime * 5.0);
            this.collisionMarker.scale.set(scale, scale, scale);
        }
        if (this.safeMarker && this.safeMarker.visible) {
            const scale = 1.0 + 0.2 * Math.sin(this.simTime * 4.0);
            this.safeMarker.scale.set(scale, scale, scale);
        }

        // Animate Satellite 1 along its orbit
        if (this.sat1Mesh) {
            const theta1 = this.simTime % (Math.PI * 2);
            let rEff = this.rOrbit;

            if (this.currentStep >= 4 && this.maneuverApplied) {
                // Follow the deflected path: clearance centered at theta = 0
                const phaseDist = Math.min(theta1, Math.PI * 2 - theta1);
                const radialKick = 0.524 * Math.exp(-(phaseDist * phaseDist) / 1.2);
                rEff += radialKick;
            }

            const p1 = this.getOrbitPoint(1, theta1, rEff);
            this.sat1Mesh.position.copy(p1);

            // Orient spacecraft nose forward along velocity vector
            const v1 = this.getOrbitVelocity(1, theta1);
            this.sat1Mesh.lookAt(p1.clone().add(v1));
        }

        // Animate Satellite 2 along its orbit
        if (this.sat2Mesh) {
            const theta2 = this.simTime % (Math.PI * 2);
            const p2 = this.getOrbitPoint(2, theta2, this.rOrbit);
            this.sat2Mesh.position.copy(p2);

            const v2 = this.getOrbitVelocity(2, theta2);
            this.sat2Mesh.lookAt(p2.clone().add(v2));
            this.sat2Mesh.rotation.z += 0.02; // Tumbling fragment spin
        }

        // Update controls
        if (this.controls) {
            this.controls.update();
        }

        // Render
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    resizeThree() {
        const container = document.getElementById('rajesh-3d-container');
        if (!container || !this.renderer || !this.camera) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        if (width === 0 || height === 0) return;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    resetCamera() {
        if (this.camera && this.controls) {
            this.camera.position.set(11, 7, 12);
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    toggleSpeed() {
        if (this.animSpeed < 0.015) {
            this.animSpeed = 0.02;
        } else {
            this.animSpeed = 0.008;
        }
        const badge = document.getElementById('rajesh-speed-badge');
        if (badge) {
            badge.textContent = (this.animSpeed > 0.01) ? 'Fast (2x)' : 'Normal (1x)';
        }
    }
}

// Global instance attachment
window.rajeshDemo = new RajeshCollisionDemo();

// Auto-initialize when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    if (window.rajeshDemo) {
        window.rajeshDemo.init();
    }
});
