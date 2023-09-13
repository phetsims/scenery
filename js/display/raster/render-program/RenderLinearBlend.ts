// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a linear blend (essentially a chunk of a linear gradient with only a linear transition between
 * two things.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderEvaluationContext, RenderExecutionStack, RenderExecutor, RenderInstruction, RenderInstructionLocation, RenderInstructionReturn, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export enum RenderLinearBlendAccuracy {
  Accurate = 0,
  PixelCenter = 1
}

scenery.register( 'RenderLinearBlendAccuracy', RenderLinearBlendAccuracy );

export default class RenderLinearBlend extends RenderProgram {

  public readonly logic: RenderLinearBlendLogic;

  public constructor(
    public readonly scaledNormal: Vector2,
    public readonly offset: number,
    public readonly accuracy: RenderLinearBlendAccuracy,
    public readonly zero: RenderProgram,
    public readonly one: RenderProgram,
    logic?: RenderLinearBlendLogic
  ) {
    assert && assert( scaledNormal.isFinite() && scaledNormal.magnitude > 0 );
    assert && assert( isFinite( offset ) );

    super(
      [ zero, one ],
      zero.isFullyTransparent && one.isFullyTransparent,
      zero.isFullyOpaque && one.isFullyOpaque,
      false,
      false,
      accuracy === RenderLinearBlendAccuracy.Accurate
    );

    this.logic = logic || new RenderLinearBlendLogic( this.scaledNormal, this.offset, this.accuracy );
  }

  public override getName(): string {
    return 'RenderLinearBlend';
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearBlend {
    assert && assert( children.length === 2 );
    return new RenderLinearBlend( this.scaledNormal, this.offset, this.accuracy, children[ 0 ], children[ 1 ], this.logic );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    // scaledNormal dot startPoint = offset
    // scaledNormal dot endPoint = offset + 1

    // scaledNormal dot ( offset * inverseScaledNormal ) = offset
    // scaledNormal dot ( ( offset + 1 ) * inverseScaledNormal ) = offset + 1

    const beforeStartPoint = this.scaledNormal.timesScalar( this.offset / this.scaledNormal.magnitudeSquared );
    const beforeEndPoint = this.scaledNormal.timesScalar( ( this.offset + 1 ) / this.scaledNormal.magnitudeSquared );

    const afterStartPoint = transform.timesVector2( beforeStartPoint );
    const afterEndPoint = transform.timesVector2( beforeEndPoint );
    const afterDelta = afterEndPoint.minus( afterStartPoint );

    const afterNormal = afterDelta.normalized().timesScalar( 1 / afterDelta.magnitude );
    const afterOffset = afterNormal.dot( afterStartPoint );

    assert && assert( Math.abs( afterNormal.dot( afterEndPoint ) - afterOffset - 1 ) < 1e-8, 'afterNormal.dot( afterEndPoint ) - afterOffset' );

    return new RenderLinearBlend(
      afterNormal,
      afterOffset,
      this.accuracy,
      this.zero.transformed( transform ),
      this.one.transformed( transform )
    );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.scaledNormal.equals( other.scaledNormal ) &&
           this.offset === other.offset &&
           this.accuracy === other.accuracy;
  }

  public override getSimplified( children: RenderProgram[] ): RenderProgram | null {
    const zero = children[ 0 ];
    const one = children[ 1 ];

    if ( zero.isFullyTransparent && one.isFullyTransparent ) {
      return RenderColor.TRANSPARENT;
    }
    else if ( zero.equals( one ) ) {
      return zero;
    }
    else {
      return null;
    }
  }

  public override evaluate( context: RenderEvaluationContext ): Vector4 {
    if ( assert && this.accuracy === RenderLinearBlendAccuracy.Accurate ) {
      assert( context.hasCentroid() );
    }

    const t = this.logic.computeLinearValue( context );

    if ( t <= 0 ) {
      return this.zero.evaluate( context );
    }
    else if ( t >= 1 ) {
      return this.one.evaluate( context );
    }
    else {
      return RenderColor.ratioBlend(
        this.zero.evaluate( context ),
        this.one.evaluate( context ),
        t
      );
    }
  }

  public override writeInstructions( instructions: RenderInstruction[] ): void {
    const zeroLocation = new RenderInstructionLocation();
    const oneLocation = new RenderInstructionLocation();
    const blendLocation = new RenderInstructionLocation();

    instructions.push( new RenderInstructionComputeLinearValue( this.logic, zeroLocation, oneLocation, blendLocation ) );
    instructions.push( zeroLocation );
    this.zero.writeInstructions( instructions );
    instructions.push( new RenderInstructionReturn() );
    instructions.push( oneLocation );
    this.one.writeInstructions( instructions );
    instructions.push( new RenderInstructionReturn() );
    instructions.push( blendLocation );
    instructions.push( new RenderInstructionLinearBlend() );
  }

  public override serialize(): SerializedRenderLinearBlend {
    return {
      type: 'RenderLinearBlend',
      scaledNormal: [ this.scaledNormal.x, this.scaledNormal.y ],
      offset: this.offset,
      accuracy: this.accuracy,
      zero: this.zero.serialize(),
      one: this.one.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderLinearBlend ): RenderLinearBlend {
    return new RenderLinearBlend(
      new Vector2( obj.scaledNormal[ 0 ], obj.scaledNormal[ 1 ] ),
      obj.offset,
      obj.accuracy,
      RenderProgram.deserialize( obj.zero ),
      RenderProgram.deserialize( obj.one )
    );
  }
}

scenery.register( 'RenderLinearBlend', RenderLinearBlend );

export class RenderLinearBlendLogic {
  public constructor(
    public readonly scaledNormal: Vector2,
    public readonly offset: number,
    public readonly accuracy: RenderLinearBlendAccuracy
  ) {}

  public computeLinearValue(
    context: RenderEvaluationContext
  ): number {
    const dot = this.accuracy === RenderLinearBlendAccuracy.Accurate ?
                this.scaledNormal.dot( context.centroid ) :
                this.scaledNormal.x * context.getCenterX() + this.scaledNormal.y * context.getCenterY();

    return dot - this.offset;
  }
}

export class RenderInstructionComputeLinearValue extends RenderInstruction {
  public constructor(
    public readonly logic: RenderLinearBlendLogic,
    public readonly zeroLocation: RenderInstructionLocation,
    public readonly oneLocation: RenderInstructionLocation,
    public readonly blendLocation: RenderInstructionLocation
  ) {
    super();
  }

  public override execute(
    stack: RenderExecutionStack,
    context: RenderEvaluationContext,
    executor: RenderExecutor
  ): void {
    const t = this.logic.computeLinearValue( context );
    stack.pushNumber( t );

    // Queue these up to be in "reverse" order
    executor.jump( this.blendLocation );

    const hasZero = t < 1;
    const hasOne = t > 0;

    if ( !hasZero || !hasOne ) {
      stack.pushValues( 0, 0, 0, 0 );
    }

    if ( hasZero ) {
      executor.call( this.zeroLocation );
    }

    if ( hasOne ) {
      executor.call( this.oneLocation );
    }
  }
}

const scratchZero = new Vector4( 0, 0, 0, 0 );
const scratchOne = new Vector4( 0, 0, 0, 0 );

// Takes `t` value from vector.x. If t <= 0 or t >= 1, it will only return the "top" value
export class RenderInstructionLinearBlend extends RenderInstruction {
  public override execute(
    stack: RenderExecutionStack,
    context: RenderEvaluationContext,
    executor: RenderExecutor
  ): void {
    const zeroColor = stack.popInto( scratchZero );
    const oneColor = stack.popInto( scratchOne );
    const t = stack.popNumber();

    if ( t <= 0 || t >= 1 ) {
      stack.push( zeroColor ); // If we're out of this range, the "top" value will always be this
    }
    else {
      const minusT = 1 - t;

      stack.pushValues(
        zeroColor.x * minusT + oneColor.x * t,
        zeroColor.y * minusT + oneColor.y * t,
        zeroColor.z * minusT + oneColor.z * t,
        zeroColor.w * minusT + oneColor.w * t
      );
    }
  }
}

export type SerializedRenderLinearBlend = {
  type: 'RenderLinearBlend';
  scaledNormal: number[];
  offset: number;
  accuracy: RenderLinearBlendAccuracy;
  zero: SerializedRenderProgram;
  one: SerializedRenderProgram;
};
