/**
 * [Form & Noise Atelier — Gimbal Node Suite]
 * Client-side ComfyUI HUD and Latent Flight Instruments extension.
 *
 * Theme Palette:
 * - Void Background: #0B0B0B
 * - Deep Accent / Border: #0E8A8A
 * - Instrument Teal: #35B8B8
 * - House Metal: #D45500
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FormAndNoise.Gimbal",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Match any Gimbal / Wayfinder nodes
        const isGimbalNode = (nodeData.name && (nodeData.name.startsWith("Gimbal") || nodeData.name.startsWith("Wayfinder"))) ||
                             (nodeData.category && (nodeData.category.includes("Gimbal") || nodeData.category.includes("Wayfinder")));

        if (isGimbalNode) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                // Atelier Dark Flight Instrument aesthetic
                this.color = "#0B0B0B";
                this.bgcolor = "#121A1A";

                return result;
            };
        }

        // Live telemetry display for GimbalDiagnostics
        if (nodeData.name === "GimbalDiagnostics") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const result = onExecuted?.apply(this, arguments);

                let hudWidget = this.widgets?.find(w => w.name === "flight_telemetry_hud");
                if (!hudWidget) {
                    hudWidget = this.addWidget(
                        "text",
                        "flight_telemetry_hud",
                        "Awaiting flight telemetry...",
                        () => {},
                        { multiline: true }
                    );
                    hudWidget.disabled = true;
                }

                if (message && message.flight_report) {
                    hudWidget.value = message.flight_report[0] || message.flight_report;
                }

                return result;
            };
        }

        // Dynamic 2D sample counter for Manifold Explorer & Circular Orbit
        if (nodeData.name === "WayfinderManifold_Explorer" || nodeData.name === "GimbalCircularOrbit") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                const updateSampleBadge = () => {
                    let total = 0;
                    if (nodeData.name === "WayfinderManifold_Explorer") {
                        const gx = this.widgets?.find(w => w.name === "grid_size_x")?.value || 3;
                        const gy = this.widgets?.find(w => w.name === "grid_size_y")?.value || 3;
                        total = gx * gy;
                    } else {
                        total = this.widgets?.find(w => w.name === "steps")?.value || 36;
                    }

                    let badgeWidget = this.widgets?.find(w => w.name === "total_batch_samples");
                    if (!badgeWidget) {
                        badgeWidget = this.addWidget("text", "total_batch_samples", `${total} Flight Coordinates`, () => {});
                        badgeWidget.disabled = true;
                    }
                    badgeWidget.value = `🎯 ${total} Latent Trajectory Coordinates`;
                };

                const originalOnWidgetChanged = this.onWidgetChanged;
                this.onWidgetChanged = function (name, value, old_value) {
                    originalOnWidgetChanged?.apply(this, arguments);
                    updateSampleBadge();
                };

                updateSampleBadge();
                return result;
            };
        }
    },

    async setup(app) {
        console.log("%c[Form & Noise Atelier] Gimbal Node Suite Loaded 🧭", "color: #35B8B8; font-weight: bold; font-size: 12px;");
    }
});
