// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a phong 3d reflection model
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderLight, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderPhong extends RenderProgram {
  public constructor(
    public readonly alpha: number,
    public readonly ambientColorProgram: RenderProgram,
    public readonly diffuseColorProgram: RenderProgram,
    public readonly specularColorProgram: RenderProgram,
    public readonly positionProgram: RenderProgram,
    public readonly normalProgram: RenderProgram,
    public readonly lights: RenderLight[]
  ) {
    super(
      [
        ambientColorProgram,
        diffuseColorProgram,
        specularColorProgram,
        positionProgram,
        normalProgram,
        ...lights.map( light => [ light.directionProgram, light.colorProgram ] ).flat()
      ],
      false,
      false
    );
  }

  public override getName(): string {
    return 'RenderPhong';
  }

  public override withChildren( children: RenderProgram[] ): RenderPhong {
    assert && assert( children.length >= 5 && children.length % 1 === 0 );

    const lightChildren = children.slice( 5 );

    return new RenderPhong(
      this.alpha,
      children[ 0 ], children[ 1 ], children[ 2 ], children[ 3 ], children[ 4 ],
      _.range( 0, lightChildren.length / 2 ).map( i => {
        return new RenderLight( lightChildren[ 2 * i ], lightChildren[ 2 * i + 1 ] );
      } ) );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.alpha === other.alpha;
  }

  public override getSimplified( children: RenderProgram[] ): RenderProgram | null {
    const ambientColorProgram = children[ 0 ];
    const diffuseColorProgram = children[ 1 ];
    const specularColorProgram = children[ 2 ];
    const positionProgram = children[ 3 ];
    const normalProgram = children[ 4 ];
    const lightChildren = children.slice( 5 );

    const numLights = lightChildren.length / 2;
    const lightDirectionPrograms = _.range( 0, numLights ).map( i => lightChildren[ 2 * i ] );
    const lightColorPrograms = _.range( 0, numLights ).map( i => lightChildren[ 2 * i + 1 ] );

    if (
      normalProgram.isFullyTransparent ||
      ( ambientColorProgram.isFullyTransparent && diffuseColorProgram.isFullyTransparent && specularColorProgram.isFullyTransparent )
    ) {
      return RenderColor.TRANSPARENT;
    }
    else if (
      ambientColorProgram instanceof RenderColor &&
      diffuseColorProgram instanceof RenderColor &&
      specularColorProgram instanceof RenderColor &&
      positionProgram instanceof RenderColor &&
      normalProgram instanceof RenderColor &&
      lightDirectionPrograms.every( program => program instanceof RenderColor ) &&
      lightColorPrograms.every( program => program instanceof RenderColor )
    ) {
      return new RenderColor(
        this.getPhong(
          ambientColorProgram.color,
          diffuseColorProgram.color,
          specularColorProgram.color,
          positionProgram.color,
          normalProgram.color,
          lightDirectionPrograms.map( program => ( program as RenderColor ).color ),
          lightColorPrograms.map( program => ( program as RenderColor ).color )
        )
      );
    }
    else {
      return null;
    }
  }

  public getPhong( ambientColor: Vector4, diffuseColor: Vector4, specularColor: Vector4, position: Vector4, normal: Vector4, lightDirections: Vector4[], lightColors: Vector4[] ): Vector4 {
    assert && assert( ambientColor.isFinite() );
    assert && assert( diffuseColor.isFinite() );
    assert && assert( specularColor.isFinite() );
    assert && assert( position.isFinite() );
    assert && assert( normal.isFinite() );
    assert && assert( lightDirections.every( direction => direction.isFinite() ) );
    assert && assert( lightDirections.every( direction => Math.abs( direction.magnitudeSquared - 1 ) < 1e-5 ) );
    assert && assert( lightColors.every( color => color.isFinite() ) );
    assert && assert( lightDirections.length === lightColors.length );

    const result = ambientColor.copy();

    for ( let i = 0; i < lightDirections.length; i++ ) {
      const lightDirection = lightDirections[ i ];
      const lightColor = lightColors[ i ];

      const dot = normal.dot( lightDirection );
      if ( dot > 0 ) {
        const diffuseColorContribution = lightColor.componentTimes( diffuseColor ).times( dot );
        diffuseColorContribution.w = lightColor.w * diffuseColor.w; // keep alpha
        result.add( diffuseColorContribution );

        // TODO: don't assume camera is at origin?
        const viewDirection = position.negated().normalized();
        const reflectedDirection = normal.timesScalar( 2 * dot ).minus( lightDirection );
        const specularContribution = Math.pow( reflectedDirection.dot( viewDirection ), this.alpha );
        const specularColorContribution = lightColor.componentTimes( specularColor ).times( specularContribution );
        specularColorContribution.w = lightColor.w * specularColor.w; // keep alpha
        result.add( specularColorContribution );
      }
    }

    // clamp for now
    result.x = Math.min( 1, result.x );
    result.y = Math.min( 1, result.y );
    result.z = Math.min( 1, result.z );
    result.w = Math.min( 1, result.w );

    return result;
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): Vector4 {
    const ambientColor = this.ambientColorProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    const diffuseColor = this.diffuseColorProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    const specularColor = this.specularColorProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    const position = this.positionProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    const normal = this.normalProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );

    // TODO: optimize?
    const lightDirections = _.range( 0, this.lights.length ).map( i => {
      return this.lights[ i ].directionProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    } );
    const lightColors = _.range( 0, this.lights.length ).map( i => {
      return this.lights[ i ].colorProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    } );

    return this.getPhong(
      ambientColor,
      diffuseColor,
      specularColor,
      position,
      normal,
      lightDirections,
      lightColors
    );
  }

  public override serialize(): SerializedRenderPhong {
    return {
      type: 'RenderPhong',
      children: this.children.map( child => child.serialize() ),
      alpha: this.alpha
    };
  }

  public static override deserialize( obj: SerializedRenderPhong ): RenderPhong {
    // @ts-expect-error
    return new RenderPhong( obj.alpha, ...obj.children.map( RenderProgram.deserialize ) );
  }
}

scenery.register( 'RenderPhong', RenderPhong );

export type SerializedRenderPhong = {
  type: 'RenderPhong';
  children: SerializedRenderProgram[];
  alpha: number;
};
