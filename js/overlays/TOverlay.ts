// Copyright 2022-2024, University of Colorado Boulder

/**
 * The main type interface for Display overlays
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

type TOverlay = {
  get domElement(): HTMLElement | SVGElement;
  update: () => void;
};
export default TOverlay;