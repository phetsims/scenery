// Copyright 2023, University of Colorado Boulder

/**
 * Enumeration of color spaces we'll want to convert from/to
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

export default class RenderColorSpace {
  protected constructor(
    public readonly name: string,
    public readonly isPremultiplied: boolean,
    public readonly isLinear: boolean
  ) {}

  public static readonly XYZ = new RenderColorSpace( 'XYZ', false, true );

  public static readonly xyY = new RenderColorSpace( 'xyY', false, false );

  public static readonly sRGB = new RenderColorSpace( 'srgb', false, false );
  public static readonly premultipliedSRGB = new RenderColorSpace( 'srgb', true, false );
  public static readonly linearSRGB = new RenderColorSpace( 'srgb', false, true );
  public static readonly premultipliedLinearSRGB = new RenderColorSpace( 'srgb', true, true );

  public static readonly displayP3 = new RenderColorSpace( 'display-p3', false, false );
  public static readonly premultipliedDisplayP3 = new RenderColorSpace( 'display-p3', true, false );
  public static readonly linearDisplayP3 = new RenderColorSpace( 'display-p3', false, true );
  public static readonly premultipliedLinearDisplayP3 = new RenderColorSpace( 'display-p3', true, true );

  public static readonly oklab = new RenderColorSpace( 'oklab', false, false );
  public static readonly premultipliedOklab = new RenderColorSpace( 'oklab', true, false );
  public static readonly linearOklab = new RenderColorSpace( 'oklab', false, true );
  public static readonly premultipliedLinearOklab = new RenderColorSpace( 'oklab', true, true );
}

scenery.register( 'RenderColorSpace', RenderColorSpace );
