// Copyright 2022, University of Colorado Boulder

/**
 * The main type interface for Display overlays
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type TOverlay = {
  get domElement(): HTMLElement | SVGElement;
  update: () => void;
};
export default TOverlay;
