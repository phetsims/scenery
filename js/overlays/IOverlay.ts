// Copyright 2022, University of Colorado Boulder

/**
 * The main type interface for Display overlays
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export default interface IOverlay {
  get domElement(): HTMLElement | SVGElement;
  update: () => void;
} // eslint-disable-line