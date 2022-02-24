// Copyright 2021-2022, University of Colorado Boulder

/**
 * The main type interface for Display overlays
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

interface IOverlay {
  get domElement(): HTMLElement | SVGElement;
  update: () => void;
}

export default IOverlay;
